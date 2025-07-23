import multiprocessing as mp
from pynput import keyboard
import time

class LifoQueue:
    """进程安全的有限容量LIFO队列，满时丢弃最旧数据。"""
    def __init__(self, maxsize):
        manager = mp.Manager()
        self.data = manager.list()
        self.maxsize = maxsize
        self.lock = mp.Lock()

    def put(self, item):
        with self.lock:
            self.data.insert(0, item)  # LIFO: 新数据放最前
            if len(self.data) > self.maxsize:
                self.data.pop()  # 丢弃最旧数据

    def get(self):
        with self.lock:
            if self.data:
                return self.data.pop(0)
            else:
                return None

    def empty(self):
        with self.lock:
            return len(self.data) == 0

    def full(self):
        with self.lock:
            return len(self.data) >= self.maxsize

def on_press(stop_event, key):
    if key == keyboard.Key.esc:
        stop_event.set()
        return False
    return True