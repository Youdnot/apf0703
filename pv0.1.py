#------------------------------------------------------------------------------
# This script adds a textured quad to the Unity scene in camera space.
# Press esc to stop.
# Test continues location.
# Unified stop event, using mt.Event() as UI control
#------------------------------------------------------------------------------

from utils.pv_stream import *
# import stop event in control_unity_ui module
from utils.control_unity_ui import *
from config import config_manager

hololens_config = config_manager.hololens_config
host = hololens_config.host

#------------------------------------------------------------------------------

# PV stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

while (not stop_event.is_set()):
    data = client.get_next_packet()

    # print_pv_stream_data(data)

    cv2.imshow('Video', data.payload.image)
    cv2.waitKey(1)

client.close()
listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
