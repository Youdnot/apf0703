import sys
import os

import numpy as np
import cv2

from pynput import keyboard

import time
from multiprocessing import Process, Lock, Condition, Manager, Event, set_start_method, current_process
from typing import Optional

class BoundedStack:
    """A simple process-safe bounded LIFO stack using a Manager-backed list.

    This container favors most-recent data. When full, it discards the oldest
    element (bottom of stack) to keep latency low. It supports blocking pop/peek
    semantics with an optional timeout.

    Attributes:
        maxsize: Maximum number of elements to keep. If <= 0, behaves as unbounded.
    """

    def __init__(self, maxsize: int = 2) -> None:
        self.maxsize = maxsize
        self._manager = Manager()
        self._stack = self._manager.list()  # type: ignore[var-annotated]
        self._mutex = Lock()
        self._not_empty = Condition(self._mutex)

    def push(self, item) -> None:
        """Pushes an item onto the top of the stack, dropping oldest if full.

        Args:
            item: Any picklable Python object (e.g., numpy arrays, small dicts).
        """
        with self._mutex:
            if self.maxsize > 0 and len(self._stack) >= self.maxsize:
                # Remove the oldest element to prioritize recency and reduce latency.
                self._stack.pop(0)
            self._stack.append(item)
            self._not_empty.notify()

    def pop(self, block: bool = True, timeout: Optional[float] = None):
        """Pops the most recent item from the stack.

        Args:
            block: If True, block until an item is available or timeout occurs.
            timeout: Maximum time to wait in seconds if blocking. None means wait indefinitely.

        Returns:
            The most recent item.

        Raises:
            TimeoutError: If blocking with a timeout and no item becomes available.
            RuntimeError: If non-blocking and the stack is empty.
        """
        with self._mutex:
            if not block:
                if len(self._stack) == 0:
                    raise RuntimeError("pop from empty stack (non-blocking)")
            elif timeout is None:
                while len(self._stack) == 0:
                    self._not_empty.wait()
            else:
                end_time = time.time() + timeout
                while len(self._stack) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise TimeoutError("pop timed out waiting for item")
                    self._not_empty.wait(remaining)

            return self._stack.pop()

    def peek(self, block: bool = True, timeout: Optional[float] = None):
        """Returns the most recent item without removing it.

        Args:
            block: If True, block until an item is available or timeout occurs.
            timeout: Maximum time to wait in seconds if blocking. None means wait indefinitely.

        Returns:
            The most recent item.

        Raises:
            TimeoutError: If blocking with a timeout and no item becomes available.
            RuntimeError: If non-blocking and the stack is empty.
        """
        with self._mutex:
            if not block:
                if len(self._stack) == 0:
                    raise RuntimeError("peek from empty stack (non-blocking)")
            elif timeout is None:
                while len(self._stack) == 0:
                    self._not_empty.wait()
            else:
                end_time = time.time() + timeout
                while len(self._stack) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise TimeoutError("peek timed out waiting for item")
                    self._not_empty.wait(remaining)

            return self._stack[-1]

    def empty(self) -> bool:
        with self._mutex:
            return len(self._stack) == 0

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'hl2ss'))
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities

class FrameProducer:
    """Simple producer using an instance method as the Process target.

    Attributes are kept minimal and picklable for spawn compatibility on macOS.
    """

    def __init__(
        self,
        stack: BoundedStack,
        host: str,
        calibration_path: str,
        pv_width: int,
        pv_height: int,
        pv_fps: int,
        max_depth: float,
    ) -> None:
        self.stack = stack
        self.host = host
        self.calibration_path = calibration_path
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps
        self.max_depth = max_depth

        # Start PV Subsystem
        hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Get RM Depth Long Throw calibration
        self.calibration_lt = hl2ss_3dcv.get_calibration_rm(self.calibration_path, self.host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
        uv2xy = self.calibration_lt.uv2xy
        self.xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calibration_lt.scale)

        self.xy1_o = self.xy1[:-1, :-1, :]
        self.xy1_d = self.xy1[1:, 1:, :]

        # Initialize PV intrinsics and extrinsics
        self.pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

        # Start PV and RM Depth Long Throw streams
        self.sink_pv = hl2ss_mp.stream(hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, width=self.pv_width, height=self.pv_height, framerate=self.pv_fps))
        self.sink_lt = hl2ss_mp.stream(hl2ss_lnm.rx_rm_depth_longthrow(self.host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        
        self.sink_pv.open()
        self.sink_lt.open()

    def _process_frames(self, data_pv, data_lt):
        timestamp = data_lt.timestamp
        pose = data_lt.pose

        # Preprocess frames
        depth = data_lt.payload.depth
        z = hl2ss_3dcv.rm_depth_normalize(depth, self.scale)
        color = data_pv.payload.image

        # Update PV intrinsics
        self.pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(self.pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(self.pv_intrinsics, self.pv_extrinsics)

        # Generate depth map for PV image
        lt_to_world = hl2ss_3dcv.camera_to_rignode(self.calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        world_to_pv = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics)
        pv_to_pv_image = hl2ss_3dcv.camera_to_image(color_intrinsics)

        lt_points_o = hl2ss_3dcv.rm_depth_to_points(self.xy1_o, z[:-1, :-1, :])
        world_points_o = hl2ss_3dcv.transform(lt_points_o, lt_to_world)
        pv_points_o = hl2ss_3dcv.transform(world_points_o, world_to_pv)
        pv_depth = pv_points_o[:, :, 2:]
        pv_uv_o = hl2ss_3dcv.project(pv_points_o, pv_to_pv_image)

        lt_points_d = hl2ss_3dcv.rm_depth_to_points(self.xy1_d, z[:-1, :-1, :])
        world_points_d = hl2ss_3dcv.transform(lt_points_d, lt_to_world)
        pv_uv_d = hl2ss_3dcv.project(world_points_d, world_to_pv @ pv_to_pv_image)

        pv_list_o = hl2ss_3dcv.block_to_list(pv_uv_o)
        pv_list_d = hl2ss_3dcv.block_to_list(pv_uv_d)
        pv_list_depth = hl2ss_3dcv.block_to_list(pv_depth)

        mask = (depth[:-1, :-1].reshape((-1,)) > 0)

        pv_list = np.hstack((np.floor(pv_list_o[mask, :]), np.floor(pv_list_d[mask, :]) + 1, pv_list_depth[mask]))
        pv_z = np.zeros((self.pv_height, self.pv_width), dtype=np.float32)

        for n in range(0, pv_list.shape[0]):
            u0, v0, u1, v1 = map(int, pv_list[n, :4])
            if not (0 <= u0 < self.pv_width and 0 < u1 <= self.pv_width and 0 <= v0 < self.pv_height and 0 < v1 <= self.pv_height):
                continue
            pv_z[v0:v1, u0:u1] = pv_list[n, 4]
        
        return color, pv_z, timestamp, pose
    
    def get_rgbd_frame(self):
        """
        Gets the most recent synchronized and processed RGB and Depth frames.
        Returns (None, None) if valid frames are not available.
        """
        _, data_lt = self.sink_lt.get_most_recent_frame()
        if (data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose)):
            return None, None, None, None

        _, data_pv = self.sink_pv.get_nearest(data_lt.timestamp)
        if (data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose)):
            return None, None, None, None
    
        return self._process_frames(data_pv, data_lt)
    
    def run(self):
        while True:
            rgb, pv_z, timestamp, pose = self.get_rgbd_frame()
            if rgb is None or pv_z is None:
                continue
            self.stack.push((rgb, pv_z, timestamp, pose))
            # time.sleep(0.01)

    def close(self):
        """
        Closes streams and stops the PV subsystem.
        """
        self.sink_pv.close()
        self.sink_lt.close()
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)


if __name__ == "__main__":

    frame_stack = BoundedStack(maxsize=2)

    # HoloLens address
    host = "169.254.10.1"

    # Calibration path (must exist but can be empty)
    calibration_path = './calibration'

    # Front RGB camera parameters
    pv_width = 1920
    pv_height = 1080
    pv_fps = 15

    # Maximum depth in meters
    # Depth camera range is 0.25m to 7.5m, here the max_depth is for vis effect
    max_depth = 3

    frame_producer = FrameProducer(
        stack=frame_stack,
        host=host,
        calibration_path=calibration_path,
        pv_width=pv_width,
        pv_height=pv_height,
        pv_fps=pv_fps,
        max_depth=max_depth,
    )

    p = Process(target=frame_producer.run, name="ProducerProcess")
    p.start()

    while True:
        rgb, pv_z, timestamp, pose = frame_stack.peek()
        print(f"timestamp: {timestamp}\npose: {pose}")

        cv2.imshow('RGB', rgb)
        cv2.imshow('Depth', hl2ss_3dcv.rm_depth_colormap(pv_z, max_depth))
        cv2.waitKey(1)

    p,join()
