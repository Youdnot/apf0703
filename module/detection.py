import os
# Set mirror for huggingface
# Don't seem to work
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from core.detection import *
from utils.mask_utils import *
from config import config_manager


detection_config = config_manager.detection_config

# Parameter settings
output_dir = detection_config.output_dir
prompt_text = detection_config.init_prompt_text
detection_interval = detection_config.detection_interval
# max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

os.makedirs(output_dir, exist_ok=True)

# Initialize the object tracker
tracker = IncrementalObjectTracker(
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
    sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_tiny.pt",
    device="cuda",
    prompt_text=prompt_text,
    detection_interval=detection_interval,
)
tracker.set_prompt(detection_config.final_prompt_text)


# Open the camera (or replace with local video file, e.g., cv2.VideoCapture("video.mp4"))
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("assets/walking test data.mp4")
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
        print(f"merged_bool_mask: {merged_bool_mask.shape}")
        # 将布尔掩码转换为uint8格式用于显示
        merged_mask_display = (merged_bool_mask * 255).astype(np.uint8)
        cv2.imshow("Merged Bool Mask", merged_mask_display)

        process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Inference", process_image_bgr)
        
        # 添加waitKey以确保窗口更新
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Info] Quit signal received.")
            break

        # tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
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