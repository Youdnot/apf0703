from core.ui_control import *
from utils.keyboard_utils import *

element_key = initialize_connection()

listener = keyboard.Listener(on_press=on_press)
listener.start()

import time

# while (not stop_event.is_set()):
#     time.sleep(1)  # Wait for the element to be created
#     update_position([0.05, 0.05, 0.5])

#     time.sleep(1)  # Wait for the element to be created
#     update_position([0.05, -0.05, 0.5])

#     time.sleep(1)  # Wait for the element to be created
#     update_position([-0.05, -0.05, 0.5])

#     time.sleep(1)  # Wait for the element to be created
#     update_position([-0.05, 0.05, 0.5])

#     time.sleep(1)  # Wait for the element to be created
#     update_position([0, 0, 0.5])

stop_event.wait()

# Clean up
disconnect()
listener.join()