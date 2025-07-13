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

# obstacles - 初始化为空障碍物
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
detection_interval = 20  # 每20帧检测一次，控制检测频率
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
# 全局变量用于动画和检测的协调
# 摄像头对象 - 在动画初始化时创建
cap = None
# 帧计数器 - 用于控制检测频率
frame_counter = 0
# 检测结果缓存 - 避免重复检测
last_detection_mask = np.zeros((view_width, view_height), dtype=bool)
# 退出标志
should_exit = False

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
# Visualization
# 可视化的坐标轴好像有点问题
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_xlim(-100, view_width + 100)
ax.set_ylim(-100, view_height + 100)

ax.set_title('Dynamic Artificial Potential Field with Real-time Detection')

# 绘制静态元素
anchor_plot, = ax.plot(anchor_point[0], anchor_point[1], 'go', markersize=10, label='Anchor Point')
# obstacles_plot, = ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro', markersize=8, label='Obstacles')

window_plot, = ax.plot(cur_pos[0], cur_pos[1], 'bo', markersize=8, label='Window Center')

# 添加缺失的path_plot定义
path_plot, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
window_plot, = ax.plot([], [], 'bo', markersize=8, label='Window Center')

# 绘制窗口
window_rect = patches.Rectangle(
    (cur_pos[0] - window_width/2, cur_pos[1] - window_height/2),
    window_width, window_height,
    linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7, label='Window'
)
ax.add_patch(window_rect)

# 绘制锚点影响范围
window_influence = patches.Circle((cur_pos[0], cur_pos[1]), radius=d0, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.05, label='Window Influence Range')
ax.add_patch(window_influence)

# 绘制锚点影响范围
anchor_influence = patches.Circle((anchor_point[0], anchor_point[1]), radius=d0, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.05, label='Anchor Influence Range')
ax.add_patch(anchor_influence)

test_plot, = ax.plot([0], [0], 'go', markersize=8, label='Axis Test')

# 使用imshow绘制障碍物
obstacle_plot = ax.imshow(obstacle_mask.T, origin='lower', 
                          extent=(0, view_width, 0, view_height),
                          cmap='Reds', alpha=0.5)

# 绘制吸引力箭头测试
attractive_force_vis = attractive_force * 9
repulsive_force_vis = repulsive_force * 9
attractive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], attractive_force_vis[0], attractive_force_vis[1],
                         width=5, color='blue')
repulsive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], repulsive_force_vis[0], repulsive_force_vis[1],
                         width=5, color='red')

# 添加总力箭头
total_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], total_force[0], total_force[1],
                         width=5, color='green')

#------------------------------------------------------------------------------
# 检测处理函数 - 从摄像头读取帧并执行检测
def process_detection():
    """
    执行实时检测，更新obstacle_mask
    返回：是否成功检测到障碍物
    """
    global cap, tracker, last_detection_mask, obstacle_mask, should_exit
    
    # 检查摄像头是否可用
    if cap is None or not cap.isOpened():
        print("[Warning] Camera not available for detection")
        return False
    
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("[Warning] Failed to capture frame for detection")
        return False
    
    try:
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 执行检测
        process_image = tracker.add_image(frame_rgb)
        
        if process_image is None or not isinstance(process_image, np.ndarray):
            print("[Warning] Detection failed, using previous mask")
            return False
        
        # 获取检测结果
        current_mask_dict = tracker.last_mask_dict
        if current_mask_dict is None:
            print("[Warning] No mask dictionary available")
            return False
        
        # 合并所有mask为bool数组
        merged_bool_mask = get_merged_bool_mask(current_mask_dict)
        
        # 检查mask尺寸是否匹配
        if merged_bool_mask.shape != (view_width, view_height):
            print(f"[Warning] Mask shape mismatch: {merged_bool_mask.shape} vs ({view_width}, {view_height})")
            # 尝试调整尺寸
            try:
                merged_bool_mask = cv2.resize(merged_bool_mask.astype(np.uint8), (view_width, view_height)) > 0
            except:
                print("[Error] Failed to resize mask")
                return False
        
        # 更新全局障碍物mask
        obstacle_mask = merged_bool_mask
        last_detection_mask = merged_bool_mask.copy()
        
        # 保存检测状态（可选）
        tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
        
        print(f"[Info] Detection successful, detected {np.sum(merged_bool_mask)} obstacle pixels")
        return True
        
    except Exception as e:
        print(f"[Error] Detection processing failed: {e}")
        return False

#------------------------------------------------------------------------------
# Animation
def init():
    """
    初始化动画 - 启动摄像头并设置初始状态
    """
    global cap, frame_counter, should_exit
    
    print("[Info] Initializing animation and camera...")
    
    # 启动摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open camera. Animation will run without detection.")
        cap = None
    else:
        print("[Info] Camera opened successfully")
    
    # 重置帧计数器
    frame_counter = 0
    should_exit = False
    
    # 初始化绘图元素
    path_plot.set_data([], [])
    window_plot.set_data([], [])
    
    return path_plot, window_plot

def update_plot(frame):
    """
    更新动画帧 - 集成检测和力场计算
    参数：
        frame: 动画帧数（由matplotlib自动提供）
    """
    global cur_pos, cur_vel, obstacle_mask, view_width, view_height, d0, k_att, k_rep, max_v, path_plot, window_plot, path_data
    global frame_counter, should_exit, attractive_force_arraw, repulsive_force_arraw, total_force_arraw
    
    # 检查退出标志
    if should_exit:
        plt.close(fig)
        return
    
    # 增加帧计数器
    frame_counter += 1
    
    # 按检测间隔执行检测
    if frame_counter % detection_interval == 0:
        print(f"[Frame {frame_counter}] Executing detection...")
        detection_success = process_detection()
        if not detection_success:
            print(f"[Frame {frame_counter}] Using previous obstacle mask")
    
    # 更新位置和速度 - 使用当前的obstacle_mask
    force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(
        cur_pos, cur_vel, anchor_point, obstacle_mask, view_width, view_height, 
        d0, k_att, k_rep, damping_factor, max_v, dt, path_data
    )

    # 可视化更新部分
    path = np.array(path_data)
    path_plot.set_data(path[:, 0], path[:, 1])
    window_plot.set_data([cur_pos[0]], [cur_pos[1]])

    # 更新障碍物显示
    obstacle_plot.set_data(obstacle_mask.T)

    # 更新窗口及其影响范围
    window_rect.set_xy((cur_pos[0] - window_width/2, cur_pos[1] - window_height/2))
    window_influence.center = cur_pos[0], cur_pos[1]

    # 更新力向量箭头
    attractive_force = get_attractive_force(cur_pos, anchor_point) * k_att * 9
    repulsive_force = get_repulsive_force(cur_pos, anchor_point, obstacle_mask, view_width, view_height, d0) * k_rep * 9
    
    # 移除旧的箭头
    if 'attractive_force_arraw' in globals():
        attractive_force_arraw.remove()
    if 'repulsive_force_arraw' in globals():
        repulsive_force_arraw.remove()
    if 'total_force_arraw' in globals():
        total_force_arraw.remove()

    # 绘制新的力向量箭头
    attractive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], attractive_force[0], attractive_force[1],
                         width=5, color='blue')

    repulsive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], repulsive_force[0], repulsive_force[1],
                         width=5, color='red')
                         
    total_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], 9 * force[0], 9 * force[1],
                         width=5, color='green')

    return path_plot, window_plot, attractive_force_arraw, repulsive_force_arraw, total_force_arraw

# 键盘事件处理函数 - 用于退出动画
def on_key(event):
    """
    处理键盘事件，支持退出动画
    """
    global should_exit
    if event.key == 'q' or event.key == 'escape':
        print("[Info] Quit signal received from keyboard")
        should_exit = True

# 连接键盘事件
fig.canvas.mpl_connect('key_press_event', on_key)

# 创建动画，需要重新调整可视化并封装图像的init和参数的update以传入，后续再调整
ani = animation.FuncAnimation(fig, update_plot, frames=2000, init_func=init, 
                             blit=False, interval=50, repeat=False)

# 外围图例
ax.legend(loc='upper left')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# 清理资源
print("[Info] Animation finished, cleaning up...")
if cap is not None:
    cap.release()
print("[Done] Program complete.")
