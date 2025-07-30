import queue
import time
import threading

#------------------------------------------------------------------------------
# A thread-safe queue to hold the most recent frame
frame_queue = queue.Queue(maxsize=1)
lock = threading.Lock()  # 添加共享锁

def frame_producer(client, stop_event):
    """This function runs in a separate thread and just gets frames."""
    while not stop_event.is_set():
        try:
            data = client.get_next_packet()
            frame = data.payload.image

            if frame is None:
                print("No frame received.")
                continue
            
            # 使用锁保护队列操作
            with lock:
                if not frame_queue.empty():
                    frame_queue.get_nowait()  # Discard old frame
                frame_queue.put_nowait(frame)  # Put the newest frame
        except queue.Full:
            pass  # Ignore if the queue is still full, we only want the latest
        except Exception as e:
            print(f"Error in producer thread: {e}")
            time.sleep(0.05)