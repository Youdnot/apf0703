#------------------------------------------------------------------------------
# Use RGB from pv and depth from RM Depth Long Throw to create RGBD stream.
# Base on sample_pv_depth_lt.py
# Experimental depth-to-PV RGBD generation via zero order hold.
# Press space to stop.
#------------------------------------------------------------------------------

import sys
import os

from pynput import keyboard

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'hl2ss'))
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import cv2
import numpy as np


class HoloLensRGBDStreamer:
    """
    Manages the connection, data streaming, and processing of RGB-D data from a HoloLens 2.
    """
    def __init__(self, host, calibration_path, pv_width, pv_height, pv_fps):
        self.host = host
        self.calibration_path = calibration_path
        self.pv_width = pv_width
        self.pv_height = pv_height
        self.pv_fps = pv_fps

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
        
        return color, pv_z

    def get_rgbd_frame(self):
        """
        Gets the most recent synchronized and processed RGB and Depth frames.
        Returns (None, None) if valid frames are not available.
        """
        _, data_lt = self.sink_lt.get_most_recent_frame()
        if (data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose)):
            return None, None

        _, data_pv = self.sink_pv.get_nearest(data_lt.timestamp)
        if (data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose)):
            return None, None
        
        return self._process_frames(data_pv, data_lt)

    def close(self):
        """
        Closes streams and stops the PV subsystem.
        """
        self.sink_pv.close()
        self.sink_lt.close()
        hl2ss_lnm.stop_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO)