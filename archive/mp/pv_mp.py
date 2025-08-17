from core.pv_stream import *
from utils.keyboard_utils import *
from config import config_manager

hololens_config = config_manager.hololens_config
host = hololens_config.host

#------------------------------------------------------------------------------

# pv stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

listener = keyboard.Listener(on_press=on_press)
listener.start()

#------------------------------------------------------------------------------

import time
from multiprocessing import Process, Queue, Lock, Manager, Condition
from queue import Empty, Full

class BoundedStack:
    def __init__(self, maxsize=2):
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

#------------------------------------------------------------------------------

def producer(q):
    while True:
        data = client.get_next_packet()
        print_pv_stream_data(data)
        q.push(data.payload.image)


def consumer(q):
    while True:
        image = q.pop()
        cv2.imshow('Video', image)
        cv2.waitKey(1)

#------------------------------------------------------------------------------

if __name__ == "__main__":
    q = BoundedStack()
    p = Process(target=producer, args=(q,))
    c = Process(target=consumer, args=(q,))

    p.start()
    c.start()

    p.join()
    c.join()

#------------------------------------------------------------------------------

client.close()
listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)