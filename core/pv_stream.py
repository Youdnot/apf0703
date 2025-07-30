import sys
import os

from pynput import keyboard

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'hl2ss'))
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import cv2

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
# shared = True
shared = False

# Camera parameters
# Ignored in shared mode
width     = 1920
height    = 1080
framerate = 30  # only 15 or 30 is available

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

def print_pv_stream_data(data):
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
