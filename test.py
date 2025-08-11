# workers.py
from multiprocessing import Process, Event, Queue
import time, queue

class FrameProducer:
    def __init__(self, client, out_q: Queue, stop: Event):
        self.client = client
        self.out_q = out_q
        self.stop = stop

    def run(self):
        while not self.stop.is_set():
            try:
                data = self.client.get_next_packet()
                frame = data.payload.image
                if frame is None:
                    continue
                try:
                    self.out_q.put(frame, timeout=0.05)
                except queue.Full:
                    pass
            except Exception as e:
                print(f"producer error: {e}")
                time.sleep(0.05)

class MovementConsumer:
    def __init__(self, in_q: Queue, stop: Event):
        self.in_q = in_q
        self.stop = stop

    def run(self):
        while not self.stop.is_set():
            try:
                frame = self.in_q.get(timeout=0.1)
                # TODO: 在这里做计算 / 控制
            except queue.Empty:
                pass
            except Exception as e:
                print(f"consumer error: {e}")
                time.sleep(0.05)

def start_workers(client):
    stop = Event()
    frame_q = Queue(maxsize=2)
    prod = FrameProducer(client, frame_q, stop)
    cons = MovementConsumer(frame_q, stop)
    p1 = Process(target=prod.run, name="frame_producer")
    p2 = Process(target=cons.run, name="movement_consumer")
    p1.start(); p2.start()
    return p1, p2, stop