from core.eet_stream import *
from utils.thread_utils import *
from config import config_manager

hololens_config = config_manager.hololens_config
host = hololens_config.host

#------------------------------------------------------------------------------

# eet stream
client = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
client.open()

listener = keyboard.Listener(on_press=on_press)
listener.start()

while (not stop_event.is_set()):
    data = client.get_next_packet()

    eet = data.payload

    print_eet_stream_data(data)

client.close()
listener.join()