#------------------------------------------------------------------------------
# Simplest UI control test
#------------------------------------------------------------------------------

from pynput import keyboard

import threading as mt
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'hl2ss'))

import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.10.1"

# Position in camera space (x, y, z)
position = [0, 0, 0.5]

# Rotation in camera space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Scale (x, y, z) in meters
scale = [0.05, 0.05, 1]

# Initial color
rgba = [1, 1, 1, 1]

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

display_list = hl2ss_rus.command_buffer()
display_list.begin_display_list() # Begin command sequence
display_list.remove_all() # Remove all objects that were created remotely
display_list.create_primitive(hl2ss_rus.PrimitiveType.Quad) # Create a quad, server will return its id
display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
display_list.set_local_transform(key, position, rotation, scale) # Set the local transform of the cube
display_list.set_color(key, rgba) # Set the color of the cube
display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the quad visible
display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
display_list.end_display_list() # End command sequence
ipc.push(display_list) # Send commands to server
results = ipc.pull(display_list) # Get results from server
# key = results[2] # Get the quad id, created by the 3rd command in the list
key = [x for x in results if x != 1][0] # Get the one and only non-1 id in result
print(results)
print(f'Created quad with id {key}')


stop_event.wait()

command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove(key) # Destroy quad
ipc.push(command_buffer)
results = ipc.pull(command_buffer)

ipc.close()

listener.join()
