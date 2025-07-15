import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'hl2ss'))

from pynput import keyboard

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