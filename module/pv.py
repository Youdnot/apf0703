from core.pv_stream import *
from utils.thread_utils import *
from config import config_manager

hololens_config = config_manager.hololens_config
host = hololens_config.host

#------------------------------------------------------------------------------

# pv stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

listener = keyboard.Listener(on_press=on_press)
listener.start()

while (not stop_event.is_set()):
    data = client.get_next_packet()

    # print_pv_stream_data(data)

    cv2.imshow('Video', data.payload.image)
    cv2.waitKey(1)

client.close()
listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)