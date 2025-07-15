from utils.control_unity_ui import *

element_key = initialize_connection()

listener = keyboard.Listener(on_press=on_press)
listener.start()

stop_event.wait()

# Clean up
disconnect()
listener.join()