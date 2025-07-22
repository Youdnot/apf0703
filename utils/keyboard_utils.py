from pynput import keyboard
import threading

stop_event = threading.Event()

def on_press(key):
    if (key == keyboard.Key.esc): 
        stop_event.set()
        return False
    return True