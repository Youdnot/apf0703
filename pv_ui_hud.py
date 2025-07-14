#------------------------------------------------------------------------------
# This script adds a textured quad to the Unity scene in camera space.
# Press esc to stop.
# Test continues location.
#------------------------------------------------------------------------------

from utils.control_unity_ui import *

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.31.89'

# Settings for pv stream

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Enable Shared Capture
# If another program is already using the PV camera, you can still stream it by
# enabling shared mode, however you cannot change the resolution and framerate
shared = True

# Camera parameters
# Ignored in shared mode
width     = 1920
height    = 1080
framerate = 5    # 30

# Video encoding profile and bitrate (None = default)
profile = hl2ss.VideoProfile.H265_MAIN
bitrate = None

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

# Settings for ui control

# Position in camera space (x, y, z)
position = [0, 0, 0.5]

# Rotation in camera space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Scale (x, y, z) in meters
scale_factor = 0.1
# ratio = 1920/1080
ratio = 1
scale = [ratio*scale_factor, scale_factor, 1]

# Texture file (must be jpg or png)
texture_file = 'assets/texture.jpg'
# texture_file = 'grid.png'

# Text
text = 'Test about to start!'

# Font size
font_size = 0.8

# Text color
rgba = [1, 1, 1, 1]

#------------------------------------------------------------------------------

# Global variables for connection and element management
ipc = None
element_key = None
stop_event = mt.Event()

#------------------------------------------------------------------------------

# 
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
from utils.pv_stream import *

hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
listener.open()

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

while (not listener.pressed()):
    data = client.get_next_packet()

    print(f'Frame captured at {data.timestamp}')
    print(f'Focal length: {data.payload.focal_length}')
    print(f'Principal point: {data.payload.principal_point}')
    print(f'Exposure Time: {data.payload.exposure_time}')
    print(f'Exposure Compensation: {data.payload.exposure_compensation}')
    print(f'Lens Position (Focus): {data.payload.lens_position}')
    print(f'Focus State: {data.payload.focus_state}')
    print(f'ISO Speed: {data.payload.iso_speed}')
    print(f'White Balance: {data.payload.white_balance}')
    print(f'ISO Gains: {data.payload.iso_gains}')
    print(f'White Balance Gains: {data.payload.white_balance_gains}')
    print(f'Resolution {data.payload.resolution}')
    print(f'Pose')
    print(data.pose)

    cv2.imshow('Video', data.payload.image)
    cv2.waitKey(1)

client.close()
listener.close()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


#------------------------------------------------------------------------------

stop_event.wait()

# Clean up
disconnect()
listener.join()