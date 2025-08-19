#------------------------------------------------------------------------------
# Combine test
# Experimental depth-to-PV RGBD generation via zero order hold.
# Extended eye tracking projection onto PV images using Long Throw depth for
# raycasting.
# No function changed compared to v2.
# Minimize the structure of the code.
# Press space to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
# import open3d as o3d
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
# import numba as nb

from data_utils import *

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.10.1"

# Calibration path (must exist but can be empty)
calibration_path = './calibration'

# Front RGB camera parameters
# pv_width = 1920
# pv_height = 1080
pv_width = 1280
pv_height = 720
pv_fps = 30

# Maximum depth in meters
# max_depth = 3.0

# Extended Eye Tracking parameters
eet_fps = 90

# Vis paras for eet in opencv, no longer used
# radius         = 5
# combined_color = (255, 0, 255)
# thickness      = -1

# Bug fix test
combined_point = None

# UTC offset for time stamp conversion
utc_offset = 0

#------------------------------------------------------------------------------
# rerun init

import numpy as np
import rerun as rr
# from rerun import blueprint as rrb

rr.init("Unified")
# rr.send_blueprint(blueprint)

rr.spawn()
# rr.set_time("stable_time", duration=0)

# Define /world coordinate convention to match HL2
rr.log("/world", rr.ViewCoordinates.RUB, static=True)

frame_idx = 0

#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Keyboard events ---------------------------------------------------------
    listener = hl2ss_utilities.key_listener(keyboard.Key.space)
    listener.open()

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Retrieve UTC offset for timestamp conversion
    # print('Retrieving UTC offset for timestamp conversion...')
    ipc_rc = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
    ipc_rc.open()
    utc_offset = ipc_rc.ts_get_utc_offset()
    ipc_rc.close()
    print(f'UTC offset: {utc_offset} (100-nanoseconds)')

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(calibration_path, host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    uv2xy = calibration_lt.uv2xy
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    # Prepare ray vectors for zero order hold interpolation -------------------
    xy1_o = xy1[:-1, :-1, :]
    xy1_d = xy1[1:, 1:, :]

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    # Start PV and RM Depth Long Throw streams --------------------------------
    sink_pv = hl2ss_mp.stream(hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_fps, decoded_format='bgr24'))
    sink_lt = hl2ss_mp.stream(hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    sink_eet = hl2ss_mp.stream(hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=eet_fps))

    sink_pv.open()
    sink_lt.open()
    sink_eet.open()

    # Initialize last timestamp for RM Depth Long Throw -----------------------
    last_lt_ts = 0

    vi_counter = hl2ss_utilities.framerate_counter()
    vi_counter.reset()

    # Main Loop ---------------------------------------------------------------
    while (not listener.pressed()):
        # Get PV frame and nearest (in time) RM Depth Long Throw frame --------
        _, data_pv = sink_pv.get_most_recent_frame()
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue

        _, data_lt = sink_lt.get_nearest(data_pv.timestamp)
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            continue
        
        # Get EET frame
        _, data_eet = sink_eet.get_most_recent_frame()
        if ((data_eet is None) or (not hl2ss.is_valid_pose(data_eet.pose))):
            continue
        
        # Preprocess frames ---------------------------------------------------
        color = data_pv.payload.image
        depth = data_lt.payload.depth
        z     = hl2ss_3dcv.rm_depth_normalize(depth, scale)
        eet   = data_eet.payload

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Create pinhole camera with proper image coordinate system
        # Rerun expects Y-down image coordinates, which matches OpenCV convention
        rr.log("/world/camera", 
                rr.Pinhole(
                    focal_length=data_pv.payload.focal_length,
                    principal_point=data_pv.payload.principal_point,
                    resolution=(pv_width, pv_height),
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

        # Generate depth map for PV image -------------------------------------
        lt_to_world    = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        world_to_pv    = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics)
        pv_to_pv_image = hl2ss_3dcv.camera_to_image(color_intrinsics)

        pv_uv_o, pv_uv_d, pv_depth = fast_transform_and_project(
            xy1_o, xy1_d, z, lt_to_world, world_to_pv, pv_to_pv_image
        )

        pv_list_o     = hl2ss_3dcv.block_to_list(pv_uv_o)
        pv_list_d     = hl2ss_3dcv.block_to_list(pv_uv_d)
        pv_list_depth = hl2ss_3dcv.block_to_list(pv_depth)

        # mask = (depth[:-1,:-1].reshape((-1,)) > 0)
        mask = (z[:-1, :-1] > 0).flatten()

        pv_list = np.hstack((np.floor(pv_list_o[mask, :]), np.floor(pv_list_d[mask, :]) + 1, pv_list_depth[mask]))

        pv_z = numba_zero_order_hold(pv_list, pv_height, pv_width)

        # EET raycasting ------------------------------------------------------
        if (hl2ss.is_valid_pose(data_pv.pose)):
            world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)

        rcs = None
        if ((data_lt is not None) and (hl2ss.is_valid_pose(data_lt.pose)) and (data_lt.timestamp != last_lt_ts)):
            last_lt_ts = data_lt.timestamp
            
            depth = data_lt.payload.depth

            lt_to_world  = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
            points       = hl2ss_3dcv.rm_depth_to_points(xy1, hl2ss_3dcv.rm_depth_normalize(depth, scale))
            world_points = hl2ss_3dcv.transform(points, lt_to_world)
            
            rcs = create_raycast_scene_optimized(depth, world_points)

        if (color is not None):
            if ((world_to_pv_image is not None) and (eet is not None) and (data_eet.pose is not None) and (rcs is not None)):
                if (eet.combined_ray_valid):
                    local_combined_ray = hl2ss_3dcv.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
                    combined_ray = hl2ss_3dcv.si_ray_transform(local_combined_ray, data_eet.pose)
                    d = rcs.cast_rays(combined_ray)['t_hit'].numpy()
                    if (np.isfinite(d)):
                        combined_point = hl2ss_3dcv.si_ray_to_point(combined_ray, d)
                        combined_image_point = hl2ss_3dcv.project(combined_point, world_to_pv_image)
                        # hl2ss_utilities.draw_points(color, combined_image_point.astype(np.int32), radius, combined_color, thickness)

        # FPS -----------------------------------------------------------------
        # ~5 FPS for 1920x1080
        vi_counter.increment()
        if (vi_counter.delta() >= 2.0):
            print(f'FPS: {vi_counter.get()}')
            vi_counter.reset()

        # Time test
        qpc_timestamp = data_pv.timestamp
        pv_timestamp = convert_qpc_to_datetime64(qpc_timestamp, utc_offset)
        # print(f"QPC Timestamp: {qpc_timestamp}")
        # print(f"Datetime Object: {datetime_obj}")

        # rerun data recording
        rr.set_time("frame", timestamp=pv_timestamp)
        rr.log("/world/camera/image", rr.Image(image=color, color_model="bgr").compress(jpeg_quality=10))
        rr.log("/world/sensor/depth", rr.DepthImage(depth, meter=1.0, colormap="viridis"))

        # aligned_depth = pv_z.astype(np.uint16)
        # rr.log("/world/camera/aligned_depth", rr.DepthImage(aligned_depth))
        
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
        
        frame_idx += 1

    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.close()
    sink_lt.close()
    sink_eet.close()

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.close()
