from front import *
from detection import *
from utils.mask_utils import *
from backend_ui_control import *

import cv2

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    #------------------------------------------------------------------------------

    frame_queue = Queue()
    mask_queue = Queue()

    frontend = FrontEnd(queue=frame_queue)

    #------------------------------------------------------------------------------

    # init detection
    # Parameter settings
    # output_dir = "./outputs"
    prompt_text = "hand."
    detection_interval = 20

    frame_idx = 0

    # os.makedirs(output_dir, exist_ok=True)

     # Initialize the object tracker
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_tiny.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt("obstacle. person. desk. table. chair. bin. viehicle. car. bus. truck.")

    #------------------------------------------------------------------------------

    # init ui connection

    from backend_ui_control import UIController
    ui_controller = UIController(offset='right', mask_queue=mask_queue, sequence_filename='assets/cpt_sequence.json')

    #------------------------------------------------------------------------------

    # Initialize rerun
    rr.init("Unified")
    rr.spawn(connect=False)  # this is the Viewer that each child process will connect to

    rr.connect_grpc()

    #------------------------------------------------------------------------------

    # Start process
    frame_process = Process(target=frontend.run, name="FrontEndProcess")

    frame_process.start()
    print(f"Front process started with PID: {frame_process.pid}")


    # Start UI process
    # ui_controller.init_element()
    # ui_controller.intro(countdown=5)

    ui_process = Process(target=ui_controller.run, name="UIProcess")
    ui_process.start()
    print(f"UI process started with PID: {ui_process.pid}")

    try:
        while True:
            color, pv_z, timestamp = frame_queue.get()
            # print(f"Received data - timestamp: {timestamp}")
            # cv2.imshow("Image", color)
            # cv2.imshow('Depth', hl2ss_3dcv.rm_depth_colormap(pv_z, max_depth=3.0))
            # cv2.waitKey(1)
            
            # print(f"[Frame {frame_idx}] Processing live frame...")
            process_image = tracker.add_image(color)

            if process_image is None or not isinstance(process_image, np.ndarray):
                # print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
                frame_idx += 1
                continue

            # 从tracker获取当前mask和元数据
            current_mask_dict = tracker.last_mask_dict
            
            # 合并所有mask为bool数组
            merged_bool_mask = get_merged_bool_mask_depth(current_mask_dict, pv_z)
            # print(f"merged_bool_mask: {merged_bool_mask.shape}")
            # 将布尔掩码转换为uint8格式用于显示
            grey_scale_mask = (merged_bool_mask * 255).astype(np.uint8)
            # cv2.imshow("Merged Bool Mask", grey_scale_mask)


            process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Live Inference", process_image_bgr)

            # Transpose and flip the maskto match the coordinate system
            obstacle_mask = np.flip(merged_bool_mask.T, axis=(1))

            # Push data to queue
            if mask_queue.qsize() >= 2:
                old_data = mask_queue.get()
                # print(f"Removed old data at {old_data[-1]}")
            mask_queue.put(obstacle_mask)

            now_utc = datetime.utcnow()
            utc_time = np.datetime64(now_utc, 'ns')
            # print(f"utc_time: {utc_time}")
            rr.set_time("time", timestamp=utc_time)

            rr.log("/detection/image", rr.Image(image=process_image, color_model="bgr").compress(jpeg_quality=10))
            rr.log("/detection/mask", rr.Image(image=grey_scale_mask).compress(jpeg_quality=10))   

            frame_idx += 1
        
    except KeyboardInterrupt:
        print("[Info] Interrupted by user (Ctrl+C).")


    frame_process.terminate()
    frame_process.join()
    frontend.close()
    print("FrontEnd process stopped")