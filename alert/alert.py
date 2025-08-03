#------------------------------------------------------------------------------
# This script creates a alert frame in the outer edges of the Hololens 2 FOV, using modified cubes.
# This script adds a cube to the Unity scene and animates it.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import threading as mt
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ui_control import *

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.10.1"

# Initial position in world space (x, y, z) in meters
# position = [0, 0, 1]
# position = [-0.4, 0, 1]     # left edge
# position = [-0.4, 0.2, 1]   #upper edge
# position = [0, -0.35, 1]  # center

# Initial rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Initial scale in meters
# scale = [0.1, 0.1, 0.1]
# scale = [1, 0.1, 0.05]  # wide frame
# scale = [0.1, 1, 0.05]  # long frame

# Initial color
rgba = [1, 0, 0, 1]

#------------------------------------------------------------------------------

stop_event = mt.Event()

def on_press(key):
    if (key == keyboard.Key.esc):
        stop_event.set()
        return False
    return True

listener = keyboard.Listener(on_press=on_press)
listener.start()

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

# key = 0

def clean():
    """Clean up the created cube and close the connection"""
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server

def create_cube(key, position, rotation, scale, rgba):
    '''Create a cube in the Unity scene with the specified parameters'''
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    # display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube) # Create a cube, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the sphere
    # display_list.set_world_transform(key, position, rotation, scale) # Set the world transform of the sphere
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform of the sphere
    display_list.set_color(key, rgba) # Set the color of the sphere
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the sphere visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    key = results[2] # Get the cube id, created by the 3rd command in the list
    print(f'Created cube with id {key}')
    return key

#------------------------------------------------------------------------------

left_frame_dict = {
    'position': [-0.44, 0, 1],
    'rotation': [0, 0, 0, 1],
    'scale': [0.1, 1, 0.01],
}

right_frame_dict = {
    'position': [0.44, 0, 1],
    'rotation': [0, 0, 0, 1],
    'scale': [0.1, 1, 0.01],
}

upper_frame_dict = {
    'position': [0, 0.2, 1],
    'rotation': [0, 0, 0, 1],
    'scale': [1, 0.1, 0.01],
}

lower_frame_dict = {
    'position': [0, -0.38, 1],
    'rotation': [0, 0, 0, 1],
    'scale': [1, 0.1, 0.01],
}

frame = [left_frame_dict, right_frame_dict, upper_frame_dict, lower_frame_dict]

for frame_dict in frame:
    frame_dict['rgba'] = rgba  # Use the same color for all frames

key_list = []

clean()

for frame_dict in frame:
    key = 0  # Reset key for each frame
    key = create_cube(
        key,
        frame_dict['position'],
        frame_dict['rotation'],
        frame_dict['scale'],
        frame_dict['rgba']
    )
    key_list.append(key)


# Stop and clean up
stop_event.wait()

# Clean frame UI
clean()
# 直接clean可以解决，但是如果要和显示UI混合使用的话这样不太合理

# 以下方法无法工作
# for key in key_list:
#     command_buffer = hl2ss_rus.command_buffer()
#     command_buffer.remove(key) # Destroy cube
#     ipc.push(command_buffer)
#     results = ipc.pull(command_buffer)

ipc.close()
disconnect()
listener.join()