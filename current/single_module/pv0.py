#------------------------------------------------------------------------------
# This script adds a textured quad to the Unity scene in camera space.
# Press esc to stop.
# Test continues location.
#------------------------------------------------------------------------------

from utils.pv_stream import *
from config import config_manager

hololens_config = config_manager.hololens_config
host = hololens_config.host

#------------------------------------------------------------------------------

# PV stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
listener.open()

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

while (not listener.pressed()):
    data = client.get_next_packet()

    # print_pv_stream_data(data)

    cv2.imshow('Video', data.payload.image)
    cv2.waitKey(1)

client.close()
listener.close()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
