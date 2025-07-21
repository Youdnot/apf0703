from pynput import keyboard
import threading as mt

stop_event = mt.Event()

def on_press(key):
    if (key == keyboard.Key.esc): 
        stop_event.set()
        return False
    return True