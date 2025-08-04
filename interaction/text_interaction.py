#------------------------------------------------------------------------------
# This script adds a 3D TextMeshPro object to the Unity scene.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import threading as mt
import time
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.ui_control import *

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.10.1"

# Position in world space (x, y, z) in meters
position = [0, 0, 0.5]

# Rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Text
text = 'Welcome to experiment!\nStart in 5 seconds...'

# Font size
font_size = 0.4

# Text color
rgba = [1, 1, 1, 1]

# Scale
scale = [2, 2, 1]

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

key = 0

def create_text(key, font_size, rgba, text, position, rotation, scale):
    '''Change text based on the initial object id'''
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    # display_list.remove_all() # Remove all objects that were created remotely
    # display_list.create_text() # Create text object, server will return its id
    # display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the text object
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.set_text(key, font_size, rgba, text) # Set text
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform
    # display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the text object visible
    # display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    # key = results[2] # Get the text object id, created by the 3rd command in the list

    print(f'Changed text object "{text}" with id {key}')
    return key

# Initialize background
def create_background(position=None):
    '''Create a background quad'''
    if position:
        bg_position = position
    else:
        bg_position = [0, 0, 0.55]
    bg_rotation = [0, 0, 0, 1]
    bg_scale = [0.2, 0.15, 0.01]
    bg_rgba = [0.1, 0.1, 0.8, 1]

    bg_key = 0

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube) # Create a quad, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
    display_list.set_local_transform(bg_key, bg_position, bg_rotation, bg_scale) # Set the local transform of the cube
    display_list.set_color(bg_key, bg_rgba) # Set the color of the cube
    display_list.set_active(bg_key, hl2ss_rus.ActiveState.Active) # Make the quad visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    bg_key = results[2] # Get the quad id, created by the 3rd command in the list

    return bg_key

# Read the sequence file
with open('assets/cpt_sequence.json', 'r') as f:
    sequence = json.load(f)

digits = sequence['digits']
is_target = sequence['is_target']
intervals = sequence['intervals']
metadata = sequence.get('metadata', {})

stimulus_duration = metadata.get('config', {}).get(
            'stimulus_duration_ms', 800) / 1000.0

print(f'Stimulus duration: {stimulus_duration} seconds')

# Create the text objects
# key = create_text(key, font_size, rgba, text, position, rotation, scale)
# time.sleep(2)

# text = '5'
# print('Changing text...')
# key = create_text(key, font_size, rgba, text, position, rotation, scale)


# Initialize the display
print("Initializing display...")
display_list = hl2ss_rus.command_buffer()
display_list.begin_display_list() # Begin command sequence
display_list.remove_all() # Remove all objects that were created remotely
display_list.create_text() # Create text object, server will return its id
display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the text object
display_list.set_text(key, font_size, rgba, text) # Set text
display_list.set_local_transform(key, position, rotation, scale=[1, 1, 1]) # Set the local transform
display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the text object visible
display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
display_list.end_display_list() # End command sequence
ipc.push(display_list) # Send commands to server
results = ipc.pull(display_list) # Get results from server
key = results[2] # Get the text object id, created by the 3rd command in the list

time.sleep(5)

# Adjust the text position for the sequence
position = [0.08, -0.08, 0.5]
bg_position = [position[0], position[1], position[2]+0.01]
print(f"Background position: {bg_position}")

bg_key = create_background(bg_position)

# Start the sequence
try:
    for i, (digit, target, interval) in enumerate(zip(digits, is_target, intervals)):
        target_str = "【目标】" if target else "【非目标】"
        print(f"第{i+1:2d}个刺激: {digit} {target_str}")

        text = str(digit)
        key = create_text(key, font_size, rgba, text, position, rotation, scale)
        time.sleep(stimulus_duration)
        interval_duration = interval / 1000.0
        time.sleep(interval_duration)

except KeyboardInterrupt:
    print("\n播放已停止")
        
print("\n播放完成")


# Stop and clean up
stop_event.wait()

command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove(key) # Destroy text object
command_buffer.remove(bg_key) # Destroy background quad
command_buffer.remove_all() # Remove all objects that were created remotely
ipc.push(command_buffer)
results = ipc.pull(command_buffer)

ipc.close()

listener.join()
