import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from multiprocessing import Process, Lock, Condition, Manager, Event, set_start_method, current_process


# -----------------------------------------------------------------------------
# Process-safe bounded stack
# -----------------------------------------------------------------------------

class BoundedStack:
    """A simple process-safe bounded LIFO stack using a Manager-backed list.

    This container favors most-recent data. When full, it discards the oldest
    element (bottom of stack) to keep latency low. It supports blocking pop/peek
    semantics with an optional timeout.

    Attributes:
        maxsize: Maximum number of elements to keep. If <= 0, behaves as unbounded.
    """

    def __init__(self, maxsize: int = 2) -> None:
        self.maxsize = maxsize
        self._manager = Manager()
        self._stack = self._manager.list()  # type: ignore[var-annotated]
        self._mutex = Lock()
        self._not_empty = Condition(self._mutex)

    def push(self, item) -> None:
        """Pushes an item onto the top of the stack, dropping oldest if full.

        Args:
            item: Any picklable Python object (e.g., numpy arrays, small dicts).
        """
        with self._mutex:
            if self.maxsize > 0 and len(self._stack) >= self.maxsize:
                # Remove the oldest element to prioritize recency and reduce latency.
                self._stack.pop(0)
            self._stack.append(item)
            self._not_empty.notify()

    def pop(self, block: bool = True, timeout: Optional[float] = None):
        """Pops the most recent item from the stack.

        Args:
            block: If True, block until an item is available or timeout occurs.
            timeout: Maximum time to wait in seconds if blocking. None means wait indefinitely.

        Returns:
            The most recent item.

        Raises:
            TimeoutError: If blocking with a timeout and no item becomes available.
            RuntimeError: If non-blocking and the stack is empty.
        """
        with self._mutex:
            if not block:
                if len(self._stack) == 0:
                    raise RuntimeError("pop from empty stack (non-blocking)")
            elif timeout is None:
                while len(self._stack) == 0:
                    self._not_empty.wait()
            else:
                end_time = time.time() + timeout
                while len(self._stack) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise TimeoutError("pop timed out waiting for item")
                    self._not_empty.wait(remaining)

            return self._stack.pop()

    def peek(self, block: bool = True, timeout: Optional[float] = None):
        """Returns the most recent item without removing it.

        Args:
            block: If True, block until an item is available or timeout occurs.
            timeout: Maximum time to wait in seconds if blocking. None means wait indefinitely.

        Returns:
            The most recent item.

        Raises:
            TimeoutError: If blocking with a timeout and no item becomes available.
            RuntimeError: If non-blocking and the stack is empty.
        """
        with self._mutex:
            if not block:
                if len(self._stack) == 0:
                    raise RuntimeError("peek from empty stack (non-blocking)")
            elif timeout is None:
                while len(self._stack) == 0:
                    self._not_empty.wait()
            else:
                end_time = time.time() + timeout
                while len(self._stack) == 0:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        raise TimeoutError("peek timed out waiting for item")
                    self._not_empty.wait(remaining)

            return self._stack[-1]

    def empty(self) -> bool:
        with self._mutex:
            return len(self._stack) == 0


# -----------------------------------------------------------------------------
# RGBD stream wrapper (HL2 or Dummy)
# -----------------------------------------------------------------------------

try:
    from core.rgbd_stream_class import HoloLensRGBDStreamer  # type: ignore
    _HL2_AVAILABLE = True
except Exception:
    _HL2_AVAILABLE = False


@dataclass
class RGBDConfig:
    """Configuration for RGBD streaming.

    Attributes:
        host: HoloLens device hostname or IP.
        calibration_path: Path for calibration data cache.
        pv_width: PV width in pixels.
        pv_height: PV height in pixels.
        pv_fps: PV frames per second.
    """

    host: str = "127.0.0.1"
    calibration_path: str = "."
    pv_width: int = 640
    pv_height: int = 360
    pv_fps: int = 30


class DummyRGBDStreamer:
    """Lightweight dummy streamer for offline testing.

    Generates synthetic RGB and depth frames to validate the pipeline without HL2.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.dt = 1.0 / max(1, fps)
        self._frame_id = 0

    def get_rgbd_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), 1000.0, dtype=np.float32)
        # Encode a moving dot to visualize updates
        u = (self._frame_id * 5) % self.width
        v = (self._frame_id * 3) % self.height
        rgb[v : v + 2, u : u + 2] = (0, 255, 0)
        self._frame_id += 1
        time.sleep(self.dt)
        return rgb, depth

    def close(self) -> None:
        pass


# -----------------------------------------------------------------------------
# Producer process
# -----------------------------------------------------------------------------

class RGBDProducer:
    """Producer that fetches RGBD frames and pushes them to a bounded stack.

    The heavy external resources (HL2 connection) are initialized inside `run()`
    to be safe with the 'spawn' start method.

    Attributes:
        stack: A bounded LIFO stack to store most recent frames.
        config: RGBD stream configuration.
        use_dummy: If True, use DummyRGBDStreamer for offline testing.
        num_items: Optional maximum number of frames to produce for tests.
        delay_seconds: Additional delay between produced frames.
        sentinel: Sentinel object to notify downstream of completion.
    """

    def __init__(
        self,
        stack: BoundedStack,
        config: RGBDConfig,
        use_dummy: bool = False,
        num_items: Optional[int] = None,
        delay_seconds: float = 0.0,
        sentinel: Optional[object] = None,
    ) -> None:
        self.stack = stack
        self.config = config
        self.use_dummy = use_dummy or not _HL2_AVAILABLE
        self.num_items = num_items
        self.delay_seconds = delay_seconds
        self.sentinel = sentinel

    def run(self, stop_event: Optional[Event] = None) -> None:
        """Runs the producer loop until stop_event is set or num_items reached.

        Args:
            stop_event: Optional multiprocessing Event to signal graceful stop.
        """
        # Initialize streamer inside the child process
        if self.use_dummy:
            streamer = DummyRGBDStreamer(
                width=self.config.pv_width,
                height=self.config.pv_height,
                fps=self.config.pv_fps,
            )
        else:
            streamer = HoloLensRGBDStreamer(
                host=self.config.host,
                calibration_path=self.config.calibration_path,
                pv_width=self.config.pv_width,
                pv_height=self.config.pv_height,
                pv_fps=self.config.pv_fps,
            )

        produced = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                rgb, depth = streamer.get_rgbd_frame()
                if rgb is None or depth is None:
                    continue

                self.stack.push((rgb, depth))
                print(f"{current_process().name} produced RGBD frame #{produced}", flush=True)

                produced += 1
                if self.num_items is not None and produced >= self.num_items:
                    break

                if self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)
        finally:
            try:
                streamer.close()
            except Exception:
                pass

            # Notify downstream consumers to exit if a sentinel is configured
            if self.sentinel is not None:
                self.stack.push(self.sentinel)


# -----------------------------------------------------------------------------
# Simple consumer for testing
# -----------------------------------------------------------------------------

def test_consumer(stack: BoundedStack, sentinel: Optional[object] = None) -> None:
    """Simple consumer that reads from the stack and prints basic info.

    Args:
        stack: Shared bounded stack with RGBD frames.
        sentinel: Optional sentinel to signal completion.
    """
    idx = 0
    while True:
        item = stack.pop(block=True)
        if sentinel is not None and item is sentinel:
            print(f"{current_process().name} received sentinel. Exiting.", flush=True)
            break

        rgb, depth = item  # type: ignore[misc]
        print(
            f"{current_process().name} consumed frame #{idx}: rgb={tuple(rgb.shape)}, depth={tuple(depth.shape)}",
            flush=True,
        )
        idx += 1


# -----------------------------------------------------------------------------
# Minimal test harness
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Configuration: prefer environment variables if provided
    cfg = RGBDConfig(
        host=os.environ.get("HL2_HOST", "127.0.0.1"),
        calibration_path=os.environ.get("HL2_CALIB_PATH", "."),
        pv_width=int(os.environ.get("HL2_PV_WIDTH", "640")),
        pv_height=int(os.environ.get("HL2_PV_HEIGHT", "360")),
        pv_fps=int(os.environ.get("HL2_PV_FPS", "30")),
    )

    # Shared bounded stack for frames
    frame_stack = BoundedStack(maxsize=2)

    # Sentinel to close the consumer after finite test production
    SENTINEL = object()

    # Producer: use dummy in offline environments; produce 8 frames and exit
    producer = RGBDProducer(
        stack=frame_stack,
        config=cfg,
        use_dummy=True,  # set to False to use actual HL2 streamer if available
        num_items=8,
        delay_seconds=0.05,
        sentinel=SENTINEL,
    )

    stop_evt = Event()

    p = Process(target=producer.run, args=(stop_evt,), name="RGBDProducer")
    c = Process(target=test_consumer, args=(frame_stack, SENTINEL), name="RGBDConsumer")

    p.start()
    c.start()

    p.join()
    c.join()

    print("Main finished.", flush=True)