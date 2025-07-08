#------------------------------------------------------------------------------
# This script adds a textured quad to the Unity scene in camera space.
# Press esc to stop.
# Test continues location.
#------------------------------------------------------------------------------

from pynput import keyboard

import threading as mt
import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.1.7'

# Position in camera space (x, y, z)
position = [0,0, 0.5]

# Rotation in camera space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Scale (x, y, z) in meters
scale = [0.05, 0.05, 1]

# Texture file (must be jpg or png)
texture_file = 'texture.jpg'

#------------------------------------------------------------------------------

# Global variables for connection and element management
ipc = None
element_key = None
stop_event = mt.Event()

def initialize_connection():
    """Initialize connection to HoloLens and create the element"""
    global ipc, element_key
    
    # Open connection
    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()
    
    # Load texture
    with open(texture_file, mode='rb') as file:
        texture = file.read()
    
    # Create element
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Quad) # Create a quad, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
    display_list.set_local_transform(0, position, rotation, scale) # Set the local transform of the cube
    display_list.set_texture(0, texture) # Set the texture of the quad
    display_list.set_active(0, hl2ss_rus.ActiveState.Active) # Make the quad visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    element_key = results[2] # Get the quad id, created by the 3rd command in the list
    
    return element_key

def update_position(new_position):
    """Update the position of the element"""
    global ipc, element_key
    
    if ipc is None or element_key is None:
        print("Connection not initialized")
        return False
    
    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.set_local_transform(element_key, new_position, rotation, scale)
    ipc.push(command_buffer)
    # results = ipc.pull(command_buffer)
    return True

def disconnect():
    """Disconnect from HoloLens and clean up"""
    global ipc, element_key
    
    if ipc is not None and element_key is not None:
        command_buffer = hl2ss_rus.command_buffer()
        command_buffer.remove(element_key) # Destroy quad
        ipc.push(command_buffer)
        # results = ipc.pull(command_buffer)
    
    if ipc is not None:
        ipc.close()
        ipc = None
        element_key = None

def on_press(key):
    if (key == keyboard.Key.esc): 
        stop_event.set()
        return False
    return True

#------------------------------------------------------------------------------

# Initialize connection and create element
element_key = initialize_connection()

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Example: Update position every 2 seconds
# You can call update_position([x, y, z]) to change position dynamically
# Example: update_position([0.1, 0.1, 0.6])

stop_event.wait()

# Clean up
disconnect()
listener.join()
