import rerun as rr
import multiprocessing
import os
import threading

@rr.shutdown_at_exit
def task(child_index: int) -> None:
    rr.init("rerun_example_multiprocessing")

    rr.connect_grpc()

    title = f"task_{child_index}"
    rr.log(
        "log",
        rr.TextLog(
            f"Logging from pid={os.getpid()}, thread={threading.get_ident()} using the Rerun recording id {rr.get_recording_id()}"
        )
    )
    if child_index == 0:
        rr.log(title, rr.Boxes2D(array=[5, 5, 80, 80], array_format=rr.Box2DFormat.XYWH, labels=title))
    else:
        rr.log(
            title,
            rr.Boxes2D(
                array=[10 + child_index * 10, 20 + child_index * 5, 30, 40],
                array_format=rr.Box2DFormat.XYWH,
                labels=title,
            ),
        )
    # rr.rerun_shutdown()

def main() -> None:
#    multiprocessing.set_start_method("spawn", force=True)
   # … existing code …

   rr.init("rerun_example_multiprocessing")
   rr.spawn(connect=False)  # this is the Viewer that each child process will connect to

   task(0)

   for i in [1, 2, 3]:
       p = multiprocessing.Process(target=task, args=(i,))
       p.start()
       p.join()

if __name__ == "__main__":
    main()