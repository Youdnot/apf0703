#------------------------------------------------------------------------------
# Experimental depth-to-PV RGBD generation via zero order hold.
# Press space to stop.
#------------------------------------------------------------------------------

# from pynput import keyboard

import numpy as np

from core.rgbd_stream import *

# Settings --------------------------------------------------------------------

# HoloLens address
host = "169.254.10.1"

# Calibration path (must exist but can be empty)
calibration_path = './calibration'

# Front RGB camera parameters
pv_width = 1920
pv_height = 1080
pv_fps = 15

# Maximum depth in meters
max_depth = 7.5

#------------------------------------------------------------------------------

if __name__ == '__main__':
    # Keyboard events
    listener = hl2ss_utilities.key_listener(keyboard.Key.space)
    listener.open()

    # Create streamer
    streamer = HoloLensRGBDStreamer(host, calibration_path, pv_width, pv_height, pv_fps)

    # Create windows
    cv2.namedWindow('RGB')
    cv2.namedWindow('Depth')

    # Framerate counter
    vi_counter = hl2ss_utilities.framerate_counter()
    vi_counter.reset()

    # Main Loop
    while not listener.pressed():
        cv2.waitKey(1)

        # Get processed frames
        color, pv_z = streamer.get_rgbd_frame()

        if color is None or pv_z is None:
            continue

        # Display RGBD pair
        cv2.imshow('RGB', color)
        cv2.imshow('Depth', hl2ss_3dcv.rm_depth_colormap(pv_z, max_depth))
        print(f'pv_z min: {np.min(pv_z)}, max: {np.max(pv_z)}')

        # FPS
        vi_counter.increment()
        if vi_counter.delta() >= 2:
            print(f'FPS: {vi_counter.get()}')
            vi_counter.reset()

    # Stop streamer and listener
    streamer.close()
    listener.close()
    cv2.destroyAllWindows()
