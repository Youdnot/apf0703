import time
from multiprocessing import Process, Queue, Event

import random
import numpy as np
from pynput import keyboard
import json
import math

import hl2ss
import hl2ss_lnm
import hl2ss_rus

#------------------------------------------------------------------------------
# Functions

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
    # display_list.remove_all() # Remove all objects that were created remotely
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

def update_text_content(ipc, key,
                font_size, text,
                rgba, texture=None):
    """Update text"""
    '''Update text based on the initial object id'''
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.set_text(key, font_size, rgba, text) # Set text
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    print(f'Changed text object "{text}" with id {key}')

#------------------------------------------------------------------------------
def update_position(ipc, key, position, rotation, scale):
    """Update the position of the element"""
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    display_list.set_local_transform(key, position, rotation, scale)
    ipc.push(display_list)
    results = ipc.pull(display_list)
    print(f'Changed position of element with id {key}')

def destroy_element(ipc, key):
    """Destroy a UI element on HoloLens"""
    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove(key) # Destroy the element
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer) # Get results from server
    print(f'Destroyed element with id {key}')
    return results

#------------------------------------------------------------------------------

def load_sequence(filename: str):
    # Read the sequence file
    with open(filename, 'r') as f:
        sequence = json.load(f)

    digits = sequence['digits']
    is_target = sequence['is_target']
    intervals = sequence['intervals']
    metadata = sequence.get('metadata', {})

    stimulus_duration = metadata.get('config', {}).get(
                'stimulus_duration_ms', 800) / 1000.0

    print(f'Stimulus duration: {stimulus_duration} seconds')

    return digits, is_target, intervals, stimulus_duration

class AlertUI:
    """A class to manage the creation and destruction of a UI frame.

    This class encapsulates the logic for creating a rectangular frame made of
    four cube primitives on the HoloLens device. It handles the initialization,
    creation, and destruction of these UI elements.

    Attributes:
        ipc: An IPC (Inter-Process Communication) object used to send commands
             to the HoloLens.
        key_list (list): A list to store the unique identifiers (keys) of the
                         created UI elements.
        frame_dicts (list): A list of dictionaries, where each dictionary
                            contains the properties (position, rotation, scale)
                            for one side of the frame.
    """

    def __init__(self, ipc):
        """Initializes the FrameUI object.

        Args:
            ipc: The IPC client for communication with the HoloLens.
        """
        self.ipc = ipc
        self.key_list = []
        
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

        self.frame_dicts = [left_frame_dict, right_frame_dict, upper_frame_dict, lower_frame_dict]
        
        for frame_dict in self.frame_dicts:
            frame_dict['rgba'] = [1, 0, 0, 1]  # Use the same color for all frames

    def create(self):
        """Creates the frame UI elements on the HoloLens.

        This method iterates through the frame dictionaries, creates each cube
        element using the provided IPC connection, and stores the returned key.
        """
        for frame_dict in self.frame_dicts:
            key = 0  # Reset key for each frame
            key = create_element_cube(
                self.ipc, key,
                position=frame_dict['position'],
                rotation=frame_dict['rotation'],
                scale=frame_dict['scale'],
                rgba=frame_dict['rgba']
            )
            self.key_list.append(key)
            print(f'Frame created with key: {key}')

    def destroy(self):
        """Destroys the frame UI elements on the HoloLens.

        This method iterates through the stored keys and sends commands to
        destroy each UI element.
        """
        for key in self.key_list:
            results = destroy_element(self.ipc, key)
            print(f'Destroyed frame with key: {key}')
        self.key_list.clear()  # Clear the list after destruction

#------------------------------------------------------------------------------
# System

def clean_up(ipc) -> None:
    # Clean before testing
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server

def intro(ipc, countdown: int = 10) -> None:
    key = 0
    welcome_text = f'Welcome to experiment!\nStart in {countdown} seconds...'
    # Create starting text object
    intro_key = create_element_text(ipc, key,
                    font_size=0.4, text=welcome_text,
                    position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                    rgba=[1, 1, 1, 1])
    print(f'Text object created with id {intro_key}')

    # Countdown
    for i in range(countdown):
        seconds_left = countdown - i
        text = str(f"Welcome to experiment!\nStart in {seconds_left} seconds...")
        update_text(ipc, intro_key,
                    font_size=0.4, text=text,
                    position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                    rgba=[1, 1, 1, 1])
        time.sleep(1)

    update_text(ipc, intro_key,
                    font_size=0.4, text="Starting now!",
                    position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                    rgba=[1, 1, 1, 1])

    time.sleep(1)

    destroy_element(ipc, intro_key)

def init(ipc, offset):

    if offset == 'left':
        text_position = [-0.08, -0.1, 0.5]
    elif offset == 'right':
        text_position = [0.08, -0.1, 0.5]
    bg_position = [*text_position[:2], text_position[2] + 0.01]


    bg_key = create_element_quad(ipc, key=0,
                   position=bg_position, rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01],
                   rgba=[0.1, 0.1, 0.8, 1])
    
    text_key = create_element_text(ipc, key=0,
                    font_size=0.4, text=str(0),
                    position=text_position, rotation=[0, 0, 0, 1], scale=[2, 2, 1],
                    rgba=[1, 1, 1, 1])

    key_dict = {
        'bg': bg_key,
        'text': text_key,
    }

    return key_dict

def cpt(ipc, key_dict, sequence_filename):
    '''continuous performance task'''

    digits, is_target, intervals, stimulus_duration = load_sequence(sequence_filename)

    text_key = key_dict['text']
    bg_key = key_dict['bg']

    try:
        for i, (digit, target, interval) in enumerate(zip(digits, is_target, intervals)):
            target_str = "【目标】" if target else "【非目标】"
            print(f"第{i+1:2d}个刺激: {digit} {target_str}")

            text = str(digit)
            update_text_content(ipc, text_key,
                    font_size=0.4, text=text,
                    rgba=[1, 1, 1, 1])
            time.sleep(stimulus_duration)
            interval_duration = interval / 1000.0
            time.sleep(interval_duration)

    except KeyboardInterrupt:
        print("\n播放已停止")
            
    print("\n播放完成")
    # rr.log(text)
    # rr.log(time)


    return key_dict

def alert(ipc, obstacle_mask, max_frame_count=10):
    '''alert frame control'''
    frame_ui = AlertUI(ipc)

    frame_count = 0
    alert = False

    while True:
        if obstacle_mask.any() and not alert:
            frame_ui.create()
            frame_count = 0
            alert = True
        elif obstacle_mask.any() and alert:
            frame_count = 0
            alert = True
        elif not obstacle_mask.any() and alert:
            frame_count += 1

        if frame_count > max_frame_count and alert:
            frame_ui.destroy()
            frame_count = 0
            alert = False

        time.sleep(0.1)


# def adaptive_movement(ipc, key_dict, obstacle_mask):
def adaptive_movement(ipc, key_dict, mask_queue):
    '''adaptive movement
    test for process
    '''

    text_key = key_dict['text']
    bg_key = key_dict['bg']

    # test sample
    # radius = 0.1
    # steps = 100

    # for i in range(steps+10):
    #     angle = 2 * math.pi * i / steps
    #     x = radius * math.cos(angle)
    #     y = radius * math.sin(angle)
    #     update_position(ipc, bg_key,
    #                 position=[x, y-0.1, 0.51], rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01])
    #     update_position(ipc, text_key,
    #                 position=[x, y-0.1, 0.5], rotation=[0, 0, 0, 1], scale=[2, 2, 1])
        
    #     time.sleep(0.05)

    # Real one

    # apf = APFCalculator(anchor, position, velocity, mask_queue)

    # while True:
    #     text_position = apf.run()
    #     bg_position = [*text_position[:2], text_position[2] + 0.01]

    #     update_position(ipc, bg_key,
    #                 position=bg_position, rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01])
    #     update_position(ipc, text_key,
    #                 position=text_position, rotation=[0, 0, 0, 1], scale=[2, 2, 1])
        
    #     time.sleep(0.05)

def unity_to_pixel(unity_position):
    '''
    Unity position to pixel position
    basically reverse ver of apfcalculator's _convert_coordinates
    '''
    pixel_position = np.array(unity_position[:2])
    pixel_position *= 226/0.1
    pixel_position += np.array([618, 329])

    return pixel_position

from apf import APFCalculator
from multiprocessing import Process

class UIController:
    def __init__(self, offset, mask_queue, sequence_filename):
        self.host = "169.254.10.1"
        self.offset = offset

        if offset == 'left':
            self.anchor = [-0.08, -0.1, 0.5]
        elif offset == 'right':
            self.anchor = [0.08, -0.1, 0.5]

        self.key_dict = {}
        self.mask_queue = mask_queue
        self.sequence_filename = sequence_filename

        anchor_pixel = unity_to_pixel(self.anchor)

        self.calculator = APFCalculator(anchor=anchor_pixel, position=anchor_pixel, mask_queue=self.mask_queue)

        self.ipc = hl2ss_lnm.ipc_umq(self.host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
        self.ipc.open()

        # Clean before testing
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list() # Begin command sequence
        display_list.remove_all() # Remove all objects that were created remotely
        display_list.end_display_list() # End command sequence
        self.ipc.push(display_list) # Send commands to server

        self.alert_ui = AlertUI(self.ipc)

    def init_element(self):
        text_position = self.anchor
        bg_position = [*text_position[:2], text_position[2] + 0.01]

        bg_key = create_element_quad(self.ipc, key=0,
                    position=bg_position, rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01],
                    rgba=[0.1, 0.1, 0.8, 1])
        
        text_key = create_element_text(self.ipc, key=0,
                        font_size=0.4, text=str(0),
                        position=text_position, rotation=[0, 0, 0, 1], scale=[2, 2, 1],
                        rgba=[1, 1, 1, 1])
        
        self.key_dict = {
            'bg': bg_key,
            'text': text_key,
        }

    def intro(self, countdown=10):
        key = 0
        welcome_text = f'Welcome to experiment!\nStart in {countdown} seconds...'
        # Create starting text object
        intro_key = create_element_text(self.ipc, key,
                        font_size=0.4, text=welcome_text,
                        position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                        rgba=[1, 1, 1, 1])
        print(f'Text object created with id {intro_key}')

        # Countdown
        for i in range(countdown):
            seconds_left = countdown - i
            text = str(f"Welcome to experiment!\nStart in {seconds_left} seconds...")
            update_text(self.ipc, intro_key,
                        font_size=0.4, text=text,
                        position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                        rgba=[1, 1, 1, 1])
            time.sleep(1)

        update_text(self.ipc, intro_key,
                        font_size=0.4, text="Starting now!",
                        position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                        rgba=[1, 1, 1, 1])

        time.sleep(1)

        destroy_element(self.ipc, intro_key)

    def cpt(self):
        '''continuous performance task'''

        digits, is_target, intervals, stimulus_duration = load_sequence(self.sequence_filename)

        text_key = self.key_dict['text']
        # bg_key = self.key_dict['bg']

        try:
            for i, (digit, target, interval) in enumerate(zip(digits, is_target, intervals)):
                target_str = "【目标】" if target else "【非目标】"
                print(f"第{i+1:2d}个刺激: {digit} {target_str}")

                text = str(digit)
                update_text_content(self.ipc, text_key,
                        font_size=0.4, text=text,
                        rgba=[1, 1, 1, 1])
                time.sleep(stimulus_duration)
                interval_duration = interval / 1000.0
                time.sleep(interval_duration)

        except KeyboardInterrupt:
            print("\n播放已停止")
                
        print("\n播放完成")

    def alert(self, obstacle_mask, max_frame_count=10):
        '''alert frame control'''

        frame_count = 0
        alert = False

        while True:
            if obstacle_mask.any() and not alert:
                self.alert_ui.create()
                frame_count = 0
                alert = True
            elif obstacle_mask.any() and alert:
                frame_count = 0
                alert = True
            elif not obstacle_mask.any() and alert:
                frame_count += 1

            if frame_count > max_frame_count and alert:
                self.alert_ui.destroy()
                frame_count = 0
                alert = False

            time.sleep(0.1)

    def adaptive_movement(self):
        text_key = self.key_dict['text']
        bg_key = self.key_dict['bg']

        # test sample
        radius = 0.1
        steps = 100

        for i in range(steps+10):
            angle = 2 * math.pi * i / steps
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            update_position(self.ipc, bg_key,
                        position=[x, y-0.1, 0.51], rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01])
            update_position(self.ipc, text_key,
                        position=[x, y-0.1, 0.5], rotation=[0, 0, 0, 1], scale=[2, 2, 1])
            
            time.sleep(0.05)

    def run(self):
        # should init be here? 
        # self.init_element()

        text_process = Process(target=self.cpt)
        movement_process = Process(target=self.adaptive_movement)
        text_process.start()
        movement_process.start()
        text_process.join()
        movement_process.join()


    def close(self):
        # Clean before exit
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list() # Begin command sequence
        display_list.remove_all() # Remove all objects that were created remotely
        display_list.end_display_list() # End command sequence
        self.ipc.push(display_list) # Send commands to server
        
        self.ipc.close()


if __name__ == "__main__":
    # q = Queue()
    # p = Process(target=producer, args=(q,))

    # p.start()
    # p.join()

    # HoloLens address
    # host = "169.254.10.1"

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

    # clean_up(ipc)

    # key_dict = init(ipc, offset='right')

    # intro(ipc, countdown=2)

    # sequence_filename = 'assets/cpt_sequence.json'

    # # cpt only change content of text
    # p1 = Process(target=cpt, args=(ipc, key_dict, sequence_filename))
    # # adaptive movement only change position of text and bg
    # # remember to add position input for it
    # p2 = Process(target=adaptive_movement, args=(ipc, key_dict))

    # p1.start()

    # time.sleep(1)

    # p2.start()

    # p1.join()
    # p2.join()


    # clean_up(ipc)

    # # Close

    # stop_event.wait()
    # ipc.close()
    # listener.join()

    ui_controller = UIController(offset='right', mask_queue=Queue(), sequence_filename='assets/cpt_sequence.json')
    ui_controller.init_element()
    ui_controller.intro(countdown=2)
    # ui_controller.run()

    p = Process(target=ui_controller.run)
    p.start()
    p.join()

    # ui_controller.cpt()

    ui_controller.close()