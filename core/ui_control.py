#------------------------------------------------------------------------------
# This script adds a textured quad to the Unity scene in camera space.
# Press esc to stop.
# Test continues location.
# Reference: https://github.com/jdibenes/hl2ss/blob/main/viewer/unity_sample_hud.py
#------------------------------------------------------------------------------

import sys
import os

# Add external directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'hl2ss'))
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_utilities

from config import config_manager

# Settings --------------------------------------------------------------------
hololens_config = config_manager.hololens_config
host = hololens_config.host

ui_config = config_manager.ui_config

position = ui_config.position
rotation = ui_config.rotation
scale = ui_config.scale
texture_file = ui_config.texture_file

#------------------------------------------------------------------------------

# Global variables for connection and element management
ipc = None
element_key = None

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

#------------------------------------------------------------------------------

# Initialize connection and create element
# element_key = initialize_connection()

# listener = keyboard.Listener(on_press=on_press)
# listener.start()

# Example: Update position every 2 seconds
# You can call update_position([x, y, z]) to change position dynamically

# import time
# time.sleep(3)  # Wait for the element to be created
# update_position([0.05, 0.05, 0.5])

# time.sleep(3)  # Wait for the element to be created
# update_position([0.05, -0.05, 0.5])

# time.sleep(3)  # Wait for the element to be created
# update_position([-0.05, -0.05, 0.5])

# time.sleep(3)  # Wait for the element to be created
# update_position([-0.05, 0.05, 0.5])

# time.sleep(3)  # Wait for the element to be created
# update_position([0, 0, 0.5])

# stop_event.wait()

# Clean up
# disconnect()
# listener.join()