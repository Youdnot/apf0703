from front import *

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    frame_queue = Queue()

    frontend = FrontEnd(queue=frame_queue)

    # Initialize rerun
    rr.init("Front")
    rr.spawn(connect=False)  # this is the Viewer that each child process will connect to
    # rr.log("/world", rr.ViewCoordinates.RUB, static=True)


    p = Process(target=frontend.run, name="FrontEndProcess")
    p.start()
    print(f"Process started with PID: {p.pid}")

    while True:
        try:
            color, pv_z, timestamp = frame_queue.get()
            # print(f"Received data - timestamp: {timestamp}")
            cv2.imshow("Image", color)
            cv2.imshow('Depth', hl2ss_3dcv.rm_depth_colormap(pv_z, max_depth=3.0))
            cv2.waitKey(1)
        
        except KeyboardInterrupt:
            print("Interrupted by user")
            break


    p.terminate()
    p.join()
    frontend.close()
    print("FrontEnd process stopped")