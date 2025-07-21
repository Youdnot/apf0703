from core.pv_stream import *
from core.ui_control import *
from config import config_manager

hololens_config = config_manager.hololens_config
host = hololens_config.host

#------------------------------------------------------------------------------
# UI control
element_key = initialize_connection()

# PV stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)
client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

# Keyboard control
listener = keyboard.Listener(on_press=on_press)
listener.start()

while (not stop_event.is_set()):
    # get pv stream data
    try:
        data = client.get_next_packet()
    except Exception as e:
        print(f"Error getting pv stream data: {e}")
        time.sleep(1)
        continue
    # print pv stream data
    # print_pv_stream_data(data)

    # update ui element position
    # update_position([0.05, 0.05, 0.5])

    cv2.imshow('Video', data.payload.image)
    print(f"width: {data.payload.image.shape[1]}, height: {data.payload.image.shape[0]}")
    cv2.waitKey(1)

# Clean up
# close pv stream client
client.close()
# stop pv stream
hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
# disconnect ui
disconnect()
# wait for keyboard control
listener.join()