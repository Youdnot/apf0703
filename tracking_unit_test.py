import copy
import os

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from external.grounding_sam2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from external.grounding_sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from external.grounding_sam2.utils.common_utils import CommonUtils
from external.grounding_sam2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from external.grounding_sam2.utils.track_utils import sample_points_from_masks
from external.grounding_sam2.utils.video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


from utils.camera_tracking import *

import os

import cv2
import torch
from external.grounding_sam2.utils.common_utils import CommonUtils


def main():
    # Parameter settings
    output_dir = "./outputs"
    prompt_text = "hand."
    detection_interval = 20
    max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize the object tracker
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt("obstacle. person. viehicle. car. bus. truck. desk. table. chair.")


    # Open the camera (or replace with local video file, e.g., cv2.VideoCapture("video.mp4"))
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("assets/walking test data.mp4")
    if not cap.isOpened():
        print("[Error] Cannot open camera.")
        return

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

            # process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Live Inference", process_image_bgr)

            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("[Info] Quit signal received.")
            #     break

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

if __name__ == "__main__":
    main()