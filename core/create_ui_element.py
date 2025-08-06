#------------------------------------------------------------------------------
# Simplest UI control test
#------------------------------------------------------------------------------

from pynput import keyboard
from multiprocessing import Event

import time
import os
import sys

# Introduce stack and stop event
# from utils.thread_utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'hl2ss'))

import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------

# # HoloLens address
# host = "169.254.10.1"

# # Position in camera space (x, y, z)
# position = [0, 0, 0.5]

# # Rotation in camera space (x, y, z, w) as a quaternion
# rotation = [0, 0, 0, 1]

# # Scale (x, y, z) in meters
# scale = [0.05, 0.05, 1]

# # Initial color
# rgba = [1, 1, 1, 1]

# #------------------------------------------------------------------------------

# # Define stop event
# stop_event = Event()

# def on_press(key):
#     if (key == keyboard.Key.esc): 
#         stop_event.set()
#         return False
#     return True

# listener = keyboard.Listener(on_press=on_press)
# listener.start()

# ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
# ipc.open()

# key = 0

#------------------------------------------------------------------------------

def create_element_quad(ipc, key,
                    position, rotation, scale,
                    rgba, texture=None):
    """Create a UI element on HoloLens
    """

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    # display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Quad) # Create a primitive, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform of the cube
    display_list.set_color(key, rgba) # Set the color of the cube
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the quad visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    print(results)
    # key = results[2] # Get the quad id, created by the 3rd command in the list
    key = [x for x in results if x != 1][0] # Get the one and only non-1 id in result
    print(f'Created Quad with id {key}')

    return key

def create_element_cube(ipc, key,
                    position, rotation, scale,
                    rgba, texture=None):
    """Create a UI element on HoloLens
    """

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    # display_list.remove_all() # Remove all objects that were created remotely

    display_list.create_primitive(hl2ss_rus.PrimitiveType.Cube) # Create a primitive, server will return its id

    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform of the cube
    display_list.set_color(key, rgba) # Set the color of the cube
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the quad visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    print(results)
    # key = results[2] # Get the quad id, created by the 3rd command in the list
    key = [x for x in results if x != 1][0] # Get the one and only non-1 id in result
    print(f'Created Quad with id {key}')

    return key

def create_element_sphere(ipc, key,
                    position, rotation, scale,
                    rgba, texture=None):
    """Create a UI element on HoloLens
    """

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    # display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_primitive(hl2ss_rus.PrimitiveType.Sphere) # Create a primitive, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the quad
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform of the cube
    display_list.set_color(key, rgba) # Set the color of the cube
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the quad visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    print(results)
    # key = results[2] # Get the quad id, created by the 3rd command in the list
    key = [x for x in results if x != 1][0] # Get the one and only non-1 id in result
    print(f'Created Quad with id {key}')

    return key

def create_element_text(ipc, key,
                font_size, text,
                position, rotation, scale,
                rgba, texture=None):
    """Create a UI element on HoloLens
    """

    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_text() # Create text object, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the text object
    display_list.set_text(key, font_size, rgba, text) # Set text
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the text object visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    print(results)
    # key = results[2] # Get the quad id, created by the 3rd command in the list
    key = [x for x in results if x != 1][0] # Get the one and only non-1 id in result
    print(f'Created Text {text} with id {key}')

    return key

def update_text(ipc, key,
                font_size, text,
                position, rotation, scale,
                rgba, texture=None):
    """Update text"""
    '''Update text based on the initial object id'''
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.set_text(key, font_size, rgba, text) # Set text
    display_list.set_local_transform(key, position, rotation, scale) # Set the local transform
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    print(f'Changed text object "{text}" with id {key}')

#------------------------------------------------------------------------------
# def update_position(ipc, key, position):

def destroy_element(ipc, key):
    """Destroy a UI element on HoloLens"""
    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove(key) # Destroy the element
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer) # Get results from server
    print(f'Destroyed element with id {key}')
    return results

#------------------------------------------------------------------------------

# # Clean before testing
# display_list = hl2ss_rus.command_buffer()
# display_list.begin_display_list() # Begin command sequence
# display_list.remove_all() # Remove all objects that were created remotely
# display_list.end_display_list() # End command sequence
# ipc.push(display_list) # Send commands to server

#------------------------------------------------------------------------------

# # Test quad
# quad_key = create_element_quad(ipc, key,
#                    position=position, rotation=rotation, scale=scale,
#                    rgba=rgba)

# time.sleep(1)

# results = destroy_element(ipc, quad_key)

# #------------------------------------------------------------------------------

# # Test text
# text_key = create_element_text(ipc, key,
#                 font_size=0.4, text='Welcome to experiment!\nStart in 5 seconds...',
#                 position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
#                 rgba=[1, 1, 1, 1])

# # Countdown
# for i in range(6):
#     seconds_left = 5 - i
#     text = str(f"Welcome to experiment!\nStart in {seconds_left} seconds...")
#     update_text(ipc, text_key,
#                 font_size=0.4, text=text,
#                 position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
#                 rgba=[1, 1, 1, 1])
#     time.sleep(1)

# update_text(ipc, text_key,
#                 font_size=0.4, text="Starting now!",
#                 position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
#                 rgba=[1, 1, 1, 1])
# time.sleep(2)

# results = destroy_element(ipc, text_key)

# # Test background
# bg_position = [0, 0, 0.51]
# bg_key = create_element_quad(ipc, key,
#                    position=bg_position, rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01],
#                    rgba=[0.1, 0.1, 0.8, 1])

# time.sleep(3)

# results = destroy_element(ipc, bg_key)


#------------------------------------------------------------------------------
# Test cube

# class FrameUI:
#     """A class to manage the creation and destruction of a UI frame.

#     This class encapsulates the logic for creating a rectangular frame made of
#     four cube primitives on the HoloLens device. It handles the initialization,
#     creation, and destruction of these UI elements.

#     Attributes:
#         ipc: An IPC (Inter-Process Communication) object used to send commands
#              to the HoloLens.
#         key_list (list): A list to store the unique identifiers (keys) of the
#                          created UI elements.
#         frame_dicts (list): A list of dictionaries, where each dictionary
#                             contains the properties (position, rotation, scale)
#                             for one side of the frame.
#     """

#     def __init__(self, ipc):
#         """Initializes the FrameUI object.

#         Args:
#             ipc: The IPC client for communication with the HoloLens.
#         """
#         self.ipc = ipc
#         self.key_list = []
        
#         left_frame_dict = {
#             'position': [-0.44, 0, 1],
#             'rotation': [0, 0, 0, 1],
#             'scale': [0.1, 1, 0.01],
#         }

#         right_frame_dict = {
#             'position': [0.44, 0, 1],
#             'rotation': [0, 0, 0, 1],
#             'scale': [0.1, 1, 0.01],
#         }

#         upper_frame_dict = {
#             'position': [0, 0.2, 1],
#             'rotation': [0, 0, 0, 1],
#             'scale': [1, 0.1, 0.01],
#         }

#         lower_frame_dict = {
#             'position': [0, -0.38, 1],
#             'rotation': [0, 0, 0, 1],
#             'scale': [1, 0.1, 0.01],
#         }

#         self.frame_dicts = [left_frame_dict, right_frame_dict, upper_frame_dict, lower_frame_dict]
        
#         for frame_dict in self.frame_dicts:
#             frame_dict['rgba'] = [1, 0, 0, 1]  # Use the same color for all frames

#     def create(self):
#         """Creates the frame UI elements on the HoloLens.

#         This method iterates through the frame dictionaries, creates each cube
#         element using the provided IPC connection, and stores the returned key.
#         """
#         for frame_dict in self.frame_dicts:
#             key = 0  # Reset key for each frame
#             key = create_element_cube(
#                 self.ipc, key,
#                 position=frame_dict['position'],
#                 rotation=frame_dict['rotation'],
#                 scale=frame_dict['scale'],
#                 rgba=frame_dict['rgba']
#             )
#             self.key_list.append(key)
#             print(f'Frame created with key: {key}')

#     def destroy(self):
#         """Destroys the frame UI elements on the HoloLens.

#         This method iterates through the stored keys and sends commands to
#         destroy each UI element.
#         """
#         for key in self.key_list:
#             results = destroy_element(self.ipc, key)
#             print(f'Destroyed frame with key: {key}')
#         self.key_list.clear()  # Clear the list after destruction

# # Initialize and use the FrameUI class
# frame_ui = FrameUI(ipc)
# frame_ui.create()

# time.sleep(2)

# frame_ui.destroy()


# # Close

# stop_event.wait()

# ipc.close()

# listener.join()
