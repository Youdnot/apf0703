import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from utils.calculate_force import get_attractive_force, get_repulsive_force, get_total_force, update_position_and_velocity
from utils.convert_coordinate import convert_coordinates

# from scipy.spatial.distance import cdist

#------------------------------------------------------------------------------
# Setting detection environment
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

#------------------------------------------------------------------------------
# Setting window parameters
# basic view
view_width = 1920
view_height = 1080

# anchor point
anchor_point = np.array([500, 700])
# anchor_point = np.array([300, 500])


# window
window_width = 200
window_height = 200

init_pos = anchor_point.copy()
init_vel = np.array([0, 0])

path_data = [init_pos.copy()]

max_v = 20

cur_pos = init_pos.copy()
cur_vel = init_vel.copy()

# obstacles
obstacle_mask = np.zeros((view_width, view_height), dtype=bool)
# 建立一个稍有重叠但是虚假的障碍物
# obstacle_mask[500:600, 600:700] = True

obstacles = obstacle_mask

# force and potential field
k_att = zeta =  10.0  # 吸引力系数
k_rep = eta = 10.0  # 排斥力系数

# 设置阻尼系数
damping_factor = 1    # 系数越大，阻尼效应越弱

d0 = 200    # 障碍物影响范围

# 设置时间步长
dt = 0.1

cur_pos[0] -= 100
cur_pos[1] -= 50

#------------------------------------------------------------------------------
# Setting detection parameters

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
tracker.set_prompt("obstacle. person. viehicle. car. bus. truck. desk. table. chair. bin.")


#------------------------------------------------------------------------------
# Initialize force
attractive_force = get_attractive_force(cur_pos, anchor_point)
attractive_force *= k_att
print(attractive_force)

repulsive_force = get_repulsive_force(cur_pos, anchor_point, obstacle_mask, view_width, view_height, d0)
repulsive_force *= k_rep
print(repulsive_force)
total_force = attractive_force + repulsive_force
print(total_force)

#------------------------------------------------------------------------------
# Tracking

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
        obstacle_mask = get_merged_bool_mask(current_mask_dict).T
        print(f"obstacle_mask: {obstacle_mask.dtype}")

        force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, anchor_point, obstacle_mask, view_width, view_height, d0, k_att, k_rep, damping_factor, max_v, dt, path_data)

        process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Inference", process_image_bgr)
        
        # 将布尔掩码转换为uint8格式用于显示
        obstacle_mask_display = (obstacle_mask * 255).astype(np.uint8)
        cv2.imshow("Merged Bool Mask", obstacle_mask_display)
        
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
