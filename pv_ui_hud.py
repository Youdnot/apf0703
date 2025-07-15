#------------------------------------------------------------------------------
# This script adds a textured quad to the Unity scene in camera space.
# Press esc to stop.
# Test continues location.
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from utils.calculate_force import get_attractive_force, get_repulsive_force, get_total_force, update_position_and_velocity
from utils.convert_coordinate import convert_coordinates
from config import config_manager


from utils.control_unity_ui import *
from utils.pv_stream import *

# Settings for movement--------------------------------------------------------

view_config = config_manager.view_config
window_config = config_manager.window_config
physics_config = config_manager.physics_config
sim_config = config_manager.sim_config

# 初始化路径数据
path_data = [sim_config.init_pos.copy()]

# 当前位置和速度
cur_pos = sim_config.init_pos.copy()
cur_vel = sim_config.init_vel.copy()

converted_pos = np.array([0, 0, 0])

# Initialize obstacle mask
obstacle_mask = np.zeros((view_config.width, view_config.height), dtype=bool)
obstacle_mask[500:600, 600:700] = True

cur_pos[0] -= 100
cur_pos[1] -= 50

# Settings for Hololens connection----------------------------------------------

# HoloLens address
# host = '192.168.31.89'
host = '169.254.10.1'


#------------------------------------------------------------------------------
# UI control
# Global variables for connection and element management
ipc = None
element_key = None
stop_event = mt.Event()

#------------------------------------------------------------------------------

# Initialize connection and create element
element_key = initialize_connection()

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Example: Update position every 2 seconds
# You can call update_position([x, y, z]) to change position dynamically
# Example: update_position([0.1, 0.1, 0.6])

# import time
# time.sleep(3)  # Wait for the element to be created
# update_position([0.05, 0.05, 0.5])


#------------------------------------------------------------------------------

# PV stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
listener.open()

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

while (not listener.pressed()):
    data = client.get_next_packet()

    print(f'Frame captured at {data.timestamp}')
    # print(f'Focal length: {data.payload.focal_length}')
    # print(f'Principal point: {data.payload.principal_point}')
    # print(f'Exposure Time: {data.payload.exposure_time}')
    # print(f'Exposure Compensation: {data.payload.exposure_compensation}')
    # print(f'Lens Position (Focus): {data.payload.lens_position}')
    # print(f'Focus State: {data.payload.focus_state}')
    # print(f'ISO Speed: {data.payload.iso_speed}')
    # print(f'White Balance: {data.payload.white_balance}')
    # print(f'ISO Gains: {data.payload.iso_gains}')
    # print(f'White Balance Gains: {data.payload.white_balance_gains}')
    # print(f'Resolution {data.payload.resolution}')
    # print(f'Pose')
    # print(data.pose)

    cv2.imshow('Video', data.payload.image)
    cv2.waitKey(1)

    # 更新位置和速度
    force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, sim_config.anchor_point, obstacle_mask, view_config.width, view_config.height, physics_config.d0, physics_config.k_att, physics_config.k_rep, physics_config.damping_factor, physics_config.max_v, physics_config.dt, path_data)
    update_position(converted_pos)

client.close()
listener.close()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


#------------------------------------------------------------------------------

stop_event.wait()

# Clean up
disconnect()
listener.join()