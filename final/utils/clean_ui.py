import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import hl2ss
import hl2ss_lnm
import hl2ss_rus

# HoloLens address
host = "169.254.10.1"

ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

key = 0

# Clean before testing
display_list = hl2ss_rus.command_buffer()
display_list.begin_display_list() # Begin command sequence
display_list.remove_all() # Remove all objects that were created remotely
display_list.end_display_list() # End command sequence
ipc.push(display_list) # Send commands to server


ipc.close()