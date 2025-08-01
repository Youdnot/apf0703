import time
from multiprocessing import Process, Queue, Lock, Manager, Condition, Event
from queue import Empty, Full
from pynput import keyboard

stop_event = Event()

def on_press(key):
    if (key == keyboard.Key.esc): 
        stop_event.set()
        return False
    return True


if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not stop_event.is_set():
        print(stop_event.is_set())
        time.sleep(0.1)

    listener.stop()
    listener.join()