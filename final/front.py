import rerun as rr
import numpy as np
import open3d as o3d
import multiprocessing
from multiprocessing import Process, Queue
import numba as nb

import hl2ss
import hl2ss_lnm
import hl2ss_3dcv
import hl2ss_mp
import hl2ss_utilities

@nb.jit(nopython=True, cache=True, parallel=True)
def numba_zero_order_hold(pv_list, pv_height, pv_width):
    pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)
    
    for n in range(pv_list.shape[0]):
        u0 = max(0, min(int(pv_list[n, 0]), pv_width-1))
        v0 = max(0, min(int(pv_list[n, 1]), pv_height-1))
        u1 = max(0, min(int(pv_list[n, 2]), pv_width))
        v1 = max(0, min(int(pv_list[n, 3]), pv_height))
        depth = pv_list[n, 4]
        
        if depth > 0 and u1 > u0 and v1 > v0:
            for v in range(v0, v1):
                for u in range(u0, u1):
                    pv_z[v, u] = depth
    
    return pv_z

class FrontEnd:
    """Simple producer using an instance method as the Process target.

    Attributes are kept minimal and picklable for spawn compatibility on macOS.
    """

    def __init__(
        self,
        queue: Queue,
        host: str = "169.254.10.1",
        calibration_path: str = "./calibration",
        pv_width: int = 1280,
        pv_height: int = 720,
        pv_fps: int = 30,
        eet_fps: int = 90,
    ) -> None:
        self.queue = queue
        self.host = host
        self.calibration_path = calibration_path
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps
        self.eet_fps = eet_fps

        # Initialize data storage
        self.utc_offset = None
        self.calibration_lt = None
        self.xy1 = None
        self.scale = None
        self.xy1_o = None
        self.xy1_d = None
        self.pv_intrinsics = None
        self.pv_extrinsics = None
        self.color_intrinsics = None
        self.color_extrinsics = None

        self.sink_pv = None
        self.sink_lt = None
        self.sink_eet = None

        self.last_lt_ts = 0
        self.frame_idx = 0
        self.vi_counter = None

    def initialize_streams(self):
        """Initialize streams and calibration data. 
        Called in the worker process.
        This is because multiprocessing do not allow the 'weakref' stream function in hl2ss to start in init.
        """
        # Start PV Subsystem
        hl2ss_lnm.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)

        # Get UTC offset for timestamp conversion
        ipc_rc = hl2ss_lnm.ipc_rc(self.host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
        ipc_rc.open()
        self.utc_offset = ipc_rc.ts_get_utc_offset()
        ipc_rc.close()
        print(f'UTC offset: {self.utc_offset} (100-nanoseconds)')

        # Get RM Depth Long Throw calibration
        # Calibration data will be downloaded if it's not in the calibration folder
        self.calibration_lt = hl2ss_3dcv.get_calibration_rm(self.calibration_path, self.host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

        uv2xy = self.calibration_lt.uv2xy
        self.xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calibration_lt.scale)

        # Prepare ray vectors for zero order hold interpolation
        self.xy1_o = self.xy1[:-1, :-1, :]
        self.xy1_d = self.xy1[1:, 1:, :]

        # Initialize PV intrinsics and extrinsics
        self.pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
        self.pv_extrinsics = np.eye(4, 4, dtype=np.float32)

        # Start PV and RM Depth Long Throw streams
        self.sink_pv = hl2ss_mp.stream(hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, width=self.pv_width, height=self.pv_height, framerate=self.pv_fps, decoded_format='bgr24'))
        self.sink_lt = hl2ss_mp.stream(hl2ss_lnm.rx_rm_depth_longthrow(self.host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
        self.sink_eet = hl2ss_mp.stream(hl2ss_lnm.rx_eet(self.host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=self.eet_fps))
        
        self.sink_pv.open()
        self.sink_lt.open()
        self.sink_eet.open()

        # Initialize frame rate counter
        self.vi_counter = hl2ss_utilities.framerate_counter()
        self.vi_counter.reset()

    def _convert_qpc_to_datetime64(self, qpc_timestamp, utc_offset):
        """
        Convert a QPC timestamp to numpy.datetime64 format.
        
        Args:
            qpc_timestamp: QPC domain timestamp (in 100-nanosecond units).
            utc_offset: UTC offset (in 100-nanosecond units).
        
        Returns:
            numpy.datetime64: Converted timestamp with nanosecond precision.
        """
        # Convert QPC to Windows FILETIME (UTC)
        filetime = hl2ss.ts_qpc_to_filetime(qpc_timestamp, utc_offset)
        
        # Convert FILETIME to Unix time (in 100-nanoseconds)
        unix_hns = hl2ss.ts_filetime_to_unix_hns(filetime)
        
        # Convert to numpy.datetime64 with nanosecond precision
        # Convert 100-nanoseconds to nanosecondsimport open3d as o3d
        unix_ns = unix_hns * 100
        
        # Create numpy.datetime64 object
        datetime_obj = np.datetime64(unix_ns, 'ns')

        # Downsample
        datetime_obj = datetime_obj.astype('datetime64[ms]')
        return datetime_obj

    def _zero_order_hold(self, pv_list, pv_height, pv_width):
        """
        Unified zero-order hold implementation that combines the best of both approaches.
        
        Args:
            pv_list: numpy array with shape (N, 5) containing [u0, v0, u1, v1, depth]
            pv_height: height of PV image
            pv_width: width of PV image
        
        Returns:
            pv_z: depth map for PV image
        """
        pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)
        
        if pv_list.shape[0] == 0:
            return pv_z
        
        # Extract and clip coordinates in one step (combines both approaches)
        u0 = np.clip(pv_list[:, 0].astype(np.int32), 0, pv_width-1)
        v0 = np.clip(pv_list[:, 1].astype(np.int32), 0, pv_height-1)
        u1 = np.clip(pv_list[:, 2].astype(np.int32), 0, pv_width)
        v1 = np.clip(pv_list[:, 3].astype(np.int32), 0, pv_height)
        depth_values = pv_list[:, 4]
        
        # Vectorized validity check (only check for positive depth and valid rectangles)
        valid_mask = (depth_values > 0) & (u1 > u0) & (v1 > v0)
        
        if not np.any(valid_mask):
            return pv_z
        
        # Filter to valid entries only
        u0_valid = u0[valid_mask]
        v0_valid = v0[valid_mask]
        u1_valid = u1[valid_mask]
        v1_valid = v1[valid_mask]
        depth_valid = depth_values[valid_mask]
        
        # Optimized loop - only process valid rectangles
        for i in range(len(u0_valid)):
            pv_z[v0_valid[i]:v1_valid[i], u0_valid[i]:u1_valid[i]] = depth_valid[i]
        
        return pv_z

    def _optimized_mesh_generation(self, depth):
        """
        Optimized mesh generation using numpy vectorization instead of nested for loops.
        
        Args:
            depth: depth image array
            world_points: 3D world coordinates for each depth pixel
        
        Returns:
            faces: list of triangle face indices
        """
        h, w = depth.shape[-2:]
        mask = depth > 0
        
        # Create coordinate grids using numpy
        i_coords, j_coords = np.meshgrid(np.arange(1, h), np.arange(1, w), indexing='ij')
        
        # Flatten coordinates for vectorized operations
        i_flat = i_coords.flatten()
        j_flat = j_coords.flatten()
        
        # Calculate all corner coordinates at once
        ul_i = (i_flat - 1) * w + (j_flat - 1)  # upper-left
        ur_i = (i_flat - 1) * w + j_flat        # upper-right  
        bl_i = i_flat * w + (j_flat - 1)        # bottom-left
        br_i = i_flat * w + j_flat              # bottom-right
        
        # Get validity masks for all corners
        ul_valid = mask[i_flat - 1, j_flat - 1]
        ur_valid = mask[i_flat - 1, j_flat]
        bl_valid = mask[i_flat, j_flat - 1]
        br_valid = mask[i_flat, j_flat]
        
        # Create triangles using vectorized conditions
        faces = []
        
        # Triangle 1: bottom-left, bottom-right, upper-right
        triangle1_mask = bl_valid & br_valid & ur_valid
        triangle1_indices = np.column_stack((bl_i[triangle1_mask], 
                                            br_i[triangle1_mask], 
                                            ur_i[triangle1_mask]))
        
        # Triangle 2: upper-right, upper-left, bottom-left  
        triangle2_mask = ur_valid & ul_valid & bl_valid
        triangle2_indices = np.column_stack((ur_i[triangle2_mask], 
                                            ul_i[triangle2_mask], 
                                            bl_i[triangle2_mask]))
        
        # Combine all triangles
        if len(triangle1_indices) > 0:
            faces.extend(triangle1_indices.tolist())
        if len(triangle2_indices) > 0:
            faces.extend(triangle2_indices.tolist())
        
        return faces

    def _create_raycast_scene_optimized(self, depth, world_points):
        """
        Create raycast scene with optimized mesh generation.
        """
        faces = self._optimized_mesh_generation(depth)

        vertices = o3d.core.Tensor(np.asarray(world_points.reshape(-1, 3), dtype=np.float32))
        triangles = o3d.core.Tensor(np.asarray(faces, dtype=np.int32))
        mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
        rcs = o3d.t.geometry.RaycastingScene()
        rcs.add_triangles(mesh)
        return rcs

    def get_data(self):
        """Get synchronized PV, depth, and EET data."""
        # Get PV frame and nearest (in time) RM Depth Long Throw frame
        _, data_pv = self.sink_pv.get_most_recent_frame()
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            return None, None, None

        _, data_lt = self.sink_lt.get_nearest(data_pv.timestamp)
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            return None, None, None
        # Get EET frame
        _, data_eet = self.sink_eet.get_most_recent_frame()
        if ((data_eet is None) or (not hl2ss.is_valid_pose(data_eet.pose)) or (not data_eet.payload.combined_ray_valid)):
            return None, None, None
        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        self.pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(self.pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        self.color_intrinsics, self.color_extrinsics = hl2ss_3dcv.pv_fix_calibration(self.pv_intrinsics, self.pv_extrinsics)
        return data_pv, data_lt, data_eet

    def depth_projection(self, data_pv, data_lt):
        # pv_z    = np.zeros((self.pv_height, self.pv_width), dtype=np.float32)
        pv_z = None

        # Generate depth map for PV image -------------------------------------
        z = hl2ss_3dcv.rm_depth_normalize(data_lt.payload.depth, self.scale)
        
        lt_to_world    = hl2ss_3dcv.camera_to_rignode(self.calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        world_to_pv    = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(self.color_extrinsics)
        pv_to_pv_image = hl2ss_3dcv.camera_to_image(self.color_intrinsics)

        lt_points_o    = hl2ss_3dcv.rm_depth_to_points(self.xy1_o, z[:-1, :-1, :])        
        world_points_o = hl2ss_3dcv.transform(lt_points_o, lt_to_world)
        pv_points_o    = hl2ss_3dcv.transform(world_points_o, world_to_pv)
        pv_depth       = pv_points_o[:, :, 2:]
        pv_uv_o        = hl2ss_3dcv.project(pv_points_o, pv_to_pv_image)

        lt_points_d    = hl2ss_3dcv.rm_depth_to_points(self.xy1_d, z[:-1, :-1, :])
        world_points_d = hl2ss_3dcv.transform(lt_points_d, lt_to_world)
        pv_uv_d        = hl2ss_3dcv.project(world_points_d, world_to_pv @ pv_to_pv_image)

        pv_list_o     = hl2ss_3dcv.block_to_list(pv_uv_o)
        pv_list_d     = hl2ss_3dcv.block_to_list(pv_uv_d)
        pv_list_depth = hl2ss_3dcv.block_to_list(pv_depth)

        mask = (z[:-1, :-1] > 0).flatten()

        pv_list = np.hstack((np.floor(pv_list_o[mask, :]), np.floor(pv_list_d[mask, :]) + 1, pv_list_depth[mask]))

        pv_z = numba_zero_order_hold(pv_list, self.pv_height, self.pv_width)

        pv_z = pv_z.astype(np.float16)

        return pv_z
        
    def eet_projection(self, data_pv, data_lt, data_eet):
        d = None
        combined_ray = None
        combined_point = None
        combined_image_point = None

        color = None
        world_to_pv_image = None
        if ((data_pv is not None)):
            color = data_pv.payload.image
            if (hl2ss.is_valid_pose(data_pv.pose)):
                world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(self.color_extrinsics) @ hl2ss_3dcv.camera_to_image(self.color_intrinsics)

        eet = None
        if ((data_eet is not None) and (hl2ss.is_valid_pose(data_eet.pose))):
            eet = data_eet.payload

        rcs = None
        if ((data_lt is not None) and (hl2ss.is_valid_pose(data_lt.pose)) and (data_lt.timestamp != self.last_lt_ts)):
            self.last_lt_ts = data_lt.timestamp
            
            depth = data_lt.payload.depth

            lt_to_world  = hl2ss_3dcv.camera_to_rignode(self.calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
            points       = hl2ss_3dcv.rm_depth_to_points(self.xy1, hl2ss_3dcv.rm_depth_normalize(depth, self.scale))
            world_points = hl2ss_3dcv.transform(points, lt_to_world)
            rcs = self._create_raycast_scene_optimized(depth, world_points)
        if (color is not None):
            if ((world_to_pv_image is not None) and (eet is not None) and (data_eet.pose is not None) and (rcs is not None)):
                if (eet.combined_ray_valid):
                    local_combined_ray = hl2ss_3dcv.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
                    combined_ray = hl2ss_3dcv.si_ray_transform(local_combined_ray, data_eet.pose)
                    d = rcs.cast_rays(combined_ray)['t_hit'].numpy()
                    if (np.isfinite(d)):
                        combined_point = hl2ss_3dcv.si_ray_to_point(combined_ray, d)
                        combined_image_point = hl2ss_3dcv.project(combined_point, world_to_pv_image)

        return d, combined_ray, combined_point, combined_image_point
    
    @rr.shutdown_at_exit
    def log_data(self, data_pv, data_lt, pv_z, combined_point, combined_image_point, combined_ray, d) -> None:
        qpc_timestamp = data_pv.timestamp
        pv_timestamp = self._convert_qpc_to_datetime64(qpc_timestamp, self.utc_offset)

        rr.init("Unified")
        rr.connect_grpc()

        rr.log("/world", rr.ViewCoordinates.RUB, static=True)
        rr.set_time("time", timestamp=pv_timestamp)
        rr.set_time("frame", timestamp=self.frame_idx)

        rr.log("/world/camera/image", rr.Image(image=data_pv.payload.image, color_model="bgr").compress(jpeg_quality=10))
        rr.log("/world/sensor/depth", rr.DepthImage(data_lt.payload.depth, meter=1.0, colormap="viridis"))

        # about 1.6m per frame as f16 for 720p
        # rr.log("/world/camera/aligned_depth", rr.DepthImage(pv_z, meter=1.0, colormap="viridis"))
        
        if (combined_point is not None):
            rr.log("/world/camera/image/gaze_point", 
                    rr.Points2D(
                        positions=[combined_image_point],
                        colors=[(255, 0, 255)],  # 洋红色
                        radii=[8.0],  # 点的大小
                        labels=["Current Gaze"]
                    ))
        
            # Log 3D gaze point
            rr.log("/world/gaze_point_3d",
                    rr.Points3D(
                        positions=[combined_point.flatten()[:3]],
                        colors=[(255, 0, 255)],
                        radii=[0.02],
                        labels=["3D Gaze Point"]
                    ))

            # Log gaze ray
            ray_origin = combined_ray[0, 0:3]
            ray_direction = combined_ray[0, 3:6]
            ray_end = ray_origin + d[0] * ray_direction
            rr.log("/world/gaze_ray",
                    rr.LineStrips3D(
                        strips=[[ray_origin, ray_end]],
                        colors=[(255, 255, 0)],
                        labels=["Gaze Ray"]
                    ))

        # Create pinhole camera with proper image coordinate system
        # Rerun expects Y-down image coordinates, which matches OpenCV convention
        rr.log("/world/camera", 
                rr.Pinhole(
                    focal_length=data_pv.payload.focal_length,
                    principal_point=data_pv.payload.principal_point,
                    resolution=(self.pv_width, self.pv_height),
                    image_plane_distance=2.0,
                    camera_xyz=rr.ViewCoordinates.RUB
                ))
        
        # Log camera pose
        # HoloLens pose is camera-to-world, but rerun expects world-to-camera
        rr.log("/world/camera", 
                       rr.Transform3D(
                           translation=data_pv.pose[3, 0:3],
                           mat3x3=np.linalg.inv(data_pv.pose[0:3, 0:3]),
                       ))

    def run(self):
        """Main processing loop that continuously captures and processes data."""
        print("FrontEnd process started...")
        
        # Initialize streams in the worker process
        self.initialize_streams()
        
        while True:
            data_pv, data_lt, data_eet = self.get_data()
            
            # Check if data is valid before processing
            if data_pv is None or data_lt is None or data_eet is None:
                continue   
            pv_z = self.depth_projection(data_pv, data_lt)

            # Push data to queue for downstream processing
            if self.queue.qsize() >= 2:
                old_data = self.queue.get()
                # print(f"Removed old data at {old_data[-1]}")
            self.queue.put((data_pv.payload.image, pv_z, data_pv.timestamp))

            d, combined_ray, combined_point, combined_image_point = self.eet_projection(data_pv, data_lt, data_eet)
            self.log_data(data_pv, data_lt, pv_z, combined_point, combined_image_point, combined_ray, d)

            # FPS -----------------------------------------------------------------
            # ~5 FPS for 1920x1080
            self.vi_counter.increment()
            if (self.vi_counter.delta() >= 2.0):
                print(f'FPS: {self.vi_counter.get()}')
                self.vi_counter.reset()

            self.frame_idx += 1

    def close(self):
        """Clean up resources and close connections."""
        if self.sink_pv:
            self.sink_pv.close()
        if self.sink_lt:
            self.sink_lt.close()
        if self.sink_eet:
            self.sink_eet.close()

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)

# Example usage (similar to demo.py main function)
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    import cv2

    frame_queue = Queue()

    frontend = FrontEnd(queue=frame_queue)

    # Initialize rerun
    rr.init("Unified")
    rr.spawn(connect=False)  # this is the Viewer that each child process will connect to
    # rr.log("/world", rr.ViewCoordinates.RUB, static=True)


    p = Process(target=frontend.run, name="FrontEndProcess")
    p.start()
    print(f"Process started with PID: {p.pid}")

    while True:
        try:
            # color, pv_z, timestamp = frontend.queue.get()
            color, pv_z, timestamp = frame_queue.get()
            # print(f"Received data - timestamp: {timestamp}")
            # cv2.imshow("Image", color)
            # cv2.imshow('Depth', hl2ss_3dcv.rm_depth_colormap(pv_z, max_depth=3.0))
            # cv2.waitKey(1)
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            break


    p.terminate()
    p.join()
    frontend.close()
    print("FrontEnd process stopped")