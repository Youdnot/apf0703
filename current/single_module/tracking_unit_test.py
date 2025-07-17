from utils.camera_tracking import *

# Parameter settings
output_dir = "./outputs"
prompt_text = "hand."
detection_interval = 10
# max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

os.makedirs(output_dir, exist_ok=True)

# Initialize the object tracker
tracker = IncrementalObjectTracker(
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    # sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    # sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_large.pt",
    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
    sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_small.pt",
    device="cuda",
    prompt_text=prompt_text,
    detection_interval=detection_interval,
)
tracker.set_prompt("obstacle. person. viehicle. car. bus. truck. desk. table. chair. bin.")


# Open the camera (or replace with local video file, e.g., cv2.VideoCapture("video.mp4"))
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("assets/walking test data.mp4")
if not cap.isOpened():
    print("[Error] Cannot open camera.")
    exit()

print("[Info] Camera opened. Press 'q' to quit.")
frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"[Frame {frame_idx}] Processing live frame...")
        process_image = tracker.add_image(frame_rgb)

        if process_image is None or not isinstance(process_image, np.ndarray):
            print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
            frame_idx += 1
            continue

        # 从tracker获取当前mask和元数据
        current_mask_dict = tracker.last_mask_dict
        
        # 合并所有mask为bool数组
        merged_bool_mask = get_merged_bool_mask(current_mask_dict)
        
        # 打印基本信息
        print(f"[Frame {frame_idx}] 合并mask形状: {merged_bool_mask.shape}")
        print(f"[Frame {frame_idx}] 检测到的对象像素数: {np.sum(merged_bool_mask)}")
        print(f"[Frame {frame_idx}] 检测到的对象数量: {len(current_mask_dict.labels)}")

        process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Inference", process_image_bgr)
        
        # 将布尔掩码转换为uint8格式用于显示
        merged_mask_display = (merged_bool_mask * 255).astype(np.uint8)
        cv2.imshow("Merged Bool Mask", merged_mask_display)
        
        # 添加waitKey以确保窗口更新
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Info] Quit signal received.")
            break

        tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
        frame_idx += 1

        # if frame_idx >= max_frames:
        #     print(f"[Info] Reached max_frames {max_frames}. Stopping.")
        #     break
except KeyboardInterrupt:
    print("[Info] Interrupted by user (Ctrl+C).")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Live inference complete.")