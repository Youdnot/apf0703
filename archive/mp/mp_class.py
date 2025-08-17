import time
from multiprocessing import Process, Queue, set_start_method, current_process
from typing import Optional


class Producer:
    """Simple producer using an instance method as the Process target.

    Attributes are kept minimal and picklable for spawn compatibility on macOS.
    """

    def __init__(
        self,
        queue: Queue,
        num_items: int = 10,
        delay_seconds: float = 0.2,
        num_consumers: int = 1,
        sentinel: Optional[object] = None,
    ) -> None:
        self.queue = queue
        self.num_items = num_items
        self.delay_seconds = delay_seconds
        self.num_consumers = max(1, num_consumers)
        self.sentinel = sentinel

    def run(self) -> None:
        for i in range(self.num_items):
            self.queue.put(i)
            print(f"{current_process().name} produced: {i}", flush=True)
            time.sleep(self.delay_seconds)

        # Send one sentinel per consumer so each can exit
        for _ in range(self.num_consumers):
            self.queue.put(self.sentinel)


def consumer(queue: Queue, sentinel: Optional[object] = None) -> None:
    while True:
        item = queue.get()
        if item is sentinel:
            print(f"{current_process().name} received sentinel. Exiting.", flush=True)
            break
        print(f"{current_process().name} consumed: {item}", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    # Explicitly use spawn on macOS for safety
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    q = Queue()
    num_consumers = 2
    sentinel = None

    producer = Producer(
        queue=q,
        num_items=10,
        delay_seconds=0.1,
        num_consumers=num_consumers,
        sentinel=sentinel,
    )

    p = Process(target=producer.run, name="ProducerProcess")
    c1 = Process(target=consumer, args=(q, sentinel), name="Consumer-1")
    c2 = Process(target=consumer, args=(q, sentinel), name="Consumer-2")

    p.start()
    c1.start()
    c2.start()

    p.join()
    c1.join()
    c2.join()

    print("Main process finished.", flush=True)
