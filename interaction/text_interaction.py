#------------------------------------------------------------------------------
# This script adds a 3D TextMeshPro object to the Unity scene.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

from multiprocessing import Event
import time
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.create_ui_element import create_element_quad, create_element_text, update_text, destroy_element, hl2ss, hl2ss_rus, hl2ss_lnm

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

# Define stop event
stop_event = Event()

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

#------------------------------------------------------------------------------

# Create starting text object
text_key = create_element_text(ipc, key,
                font_size=0.4, text='Welcome to experiment!\nStart in 5 seconds...',
                position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                rgba=[1, 1, 1, 1])
print(f'Text object created with id {text_key}')

# Countdown
for i in range(6):
    seconds_left = 5 - i
    text = str(f"Welcome to experiment!\nStart in {seconds_left} seconds...")
    update_text(ipc, text_key,
                font_size=0.4, text=text,
                position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                rgba=[1, 1, 1, 1])
    time.sleep(1)

update_text(ipc, text_key,
                font_size=0.4, text="Starting now!",
                position=[0, 0, 0.5], rotation=[0, 0, 0, 1], scale=[1, 1, 1],
                rgba=[1, 1, 1, 1])

time.sleep(0.5)

# Adjust the text position for the sequence
position = [0.08, -0.08, 0.5]
bg_position = [position[0], position[1], position[2]+0.01]
print(f"Background position: {bg_position}")

bg_key = create_element_quad(ipc, key,
                   position=bg_position, rotation=[0, 0, 0, 1], scale=[0.2, 0.15, 0.01],
                   rgba=[1, 1, 1, 1])
                #    rgba=[0.1, 0.1, 0.8, 1])
print(f'Background created with id {bg_key}')
    
# Start the sequence
try:
    for i, (digit, target, interval) in enumerate(zip(digits, is_target, intervals)):
        target_str = "【目标】" if target else "【非目标】"
        print(f"第{i+1:2d}个刺激: {digit} {target_str}")

        text = str(digit)
        # key = update_text(text_key, font_size, rgba, text, position, rotation, scale)
        update_text(ipc, text_key,
                font_size=0.4, text=text,
                position=position, rotation=[0, 0, 0, 1], scale=[2, 2, 1],
                rgba=[0.1, 0.1, 0.8, 1])
                # rgba=[1, 1, 1, 1])
        time.sleep(stimulus_duration)
        interval_duration = interval / 1000.0
        time.sleep(interval_duration)

except KeyboardInterrupt:
    print("\n播放已停止")
        
print("\n播放完成")


# Stop and clean up
stop_event.wait()

results = destroy_element(ipc, text_key)
results = destroy_element(ipc, bg_key)
print(f'Text object with id {text_key} destroyed')
print(f'Background object with id {bg_key} destroyed')

ipc.close()

listener.join()