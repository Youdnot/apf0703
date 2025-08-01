import time
import copy
from multiprocessing import Process, Queue, Lock, Manager, Condition, Event
from queue import Empty, Full
from pynput import keyboard

# stack
class BoundedStack:
    def __init__(self, maxsize=1):
        self.maxsize = maxsize
        self.manager = Manager()
        self._stack = self.manager.list()
        self._mutex = Lock()
        self._not_empty = Condition(self._mutex)

    def push(self, item):
        # block 和 timeout 参数保留以兼容，但忽略
        with self._mutex:
            if self.maxsize > 0 and len(self._stack) >= self.maxsize:
                # 移除最旧的元素 (底部)，注意 list.pop(0) 是 O(n)，对于大 maxsize 不高效
                self._stack.pop(0)
            self._stack.append(item)
            self._not_empty.notify()

    def pop(self, block=True, timeout=None):
        with self._mutex:
            if not block:
                if len(self._stack) == 0:
                    raise Empty
            elif timeout is None:
                while len(self._stack) == 0:
                    self._not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                end = time.time() + timeout
                while len(self._stack) == 0:
                    remaining = end - time.time()
                    if remaining <= 0.0:
                        raise Empty
                    self._not_empty.wait(remaining)
            item = self._stack.pop()
            return item
        
    def peek(self, block=True, timeout=None):
        with self._mutex:
            if not block:
                if len(self._stack) == 0:
                    raise Empty
            elif timeout is None:
                while len(self._stack) == 0:
                    self._not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                end = time.time() + timeout
                while len(self._stack) == 0:
                    remaining = end - time.time()
                    if remaining <= 0.0:
                        raise Empty
                    self._not_empty.wait(remaining)
            return copy.deepcopy(self._stack[-1])
        
    def empty(self):
        with self._mutex:
            return len(self._stack) == 0

    def full(self):
        with self._mutex:
            return self.maxsize > 0 and len(self._stack) >= self.maxsize
    

# stop event
stop_event = Event()

def on_press(key):
    if (key == keyboard.Key.esc): 
        stop_event.set()
        return False
    return True

# producer for RGB frame
frame_queue = BoundedStack(maxsize=2)

def frame_producer(client, stop_event):
    """This function runs in a separate thread and just gets frames."""
    while not stop_event.is_set():
        try:
            data = client.get_next_packet()
            frame = data.payload.image

            if frame is None:
                print("No frame received.")
                continue
            
            frame_queue.push(frame)
        except Exception as e:
            print(f"Error in producer thread: {e}")
            time.sleep(0.05)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.calculate_force import *
from core.ui_control import *

# consumer for mask to movement
mask_queue = BoundedStack(maxsize=2)


def movement_consumer(sim_config, view_config, physics_config, stop_event):

    cur_pos = sim_config.init_pos.copy()
    cur_vel = sim_config.init_vel.copy()
    path_data = [sim_config.init_pos.copy()]

    while not stop_event.is_set():
        try:
            obstacle_mask = mask_queue.peek()

            force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, sim_config.anchor_point, obstacle_mask, view_config.width, view_config.height, physics_config.d0, physics_config.k_att, physics_config.k_rep, physics_config.damping_factor, physics_config.max_v, physics_config.dt, path_data)
            update_position(converted_pos)
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in movement thread: {e}")
            continue