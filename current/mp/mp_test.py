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

# def producer(q):
#     for i in range(10):
#         q.push(i)
#         print(f"Produced: {i}")
#         time.sleep(0.2)
#     q.push(None)

# def consumer(q):
#     while True:
#         item = q.pop()
#         if item is None:
#             break
#         print(f"Consumed: {item}")
#         time.sleep(1)

# if __name__ == "__main__":
#     q = BoundedStack()
#     p = Process(target=producer, args=(q,))
#     c1 = Process(target=consumer, args=(q,))
#     c2 = Process(target=consumer, args=(q,))

#     p.start()
#     c1.start()
#     c2.start()
#     p.join()
#     c1.join()
#     c2.join()