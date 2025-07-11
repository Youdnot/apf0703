import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from utils.calculate_force import get_attractive_force, get_repulsive_force, get_total_force, update_position_and_velocity
from utils.convert_coordinate import convert_coordinates

# Setting parameters
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

converted_pos = np.array([0, 0, 0])

# obstacles
# 建立一个稍有重叠但是虚假的障碍物
obstacle_mask = np.zeros((view_width, view_height), dtype=bool)
obstacle_mask[500:600, 600:700] = True

obstacles = obstacle_mask

# force and potential field
k_att = zeta =  10.0  # 吸引力系数
k_rep = eta = 10.0  # 排斥力系数

# 设置阻尼系数
damping_factor = 1    # 系数越大，阻尼效应越弱

d0 = 200    # 障碍物影响范围

# 设置时间步长
dt = 0.2

cur_pos[0] -= 100
cur_pos[1] -= 50

attractive_force = get_attractive_force(cur_pos, anchor_point)
attractive_force *= k_att
print(attractive_force)

repulsive_force = get_repulsive_force(cur_pos, anchor_point, obstacle_mask, view_width, view_height, d0)
repulsive_force *= k_rep
print(repulsive_force)
total_force = attractive_force + repulsive_force
print(total_force)

# UI control
from utils.control_unity_ui import *

# Initialize connection and create element
element_key = initialize_connection()

# 启动键盘监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

import time

# 主循环：更新位置和速度
try:
    for i in range(2000):
        # 检查停止事件是否被设置
        if stop_event.is_set():
            print("收到停止信号，退出循环")
            break
            
        # 更新位置和速度
        force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, anchor_point, obstacle_mask, view_width, view_height, d0, k_att, k_rep, damping_factor, max_v, dt, path_data)
        update_position(converted_pos)
        
        time.sleep(0.1)  # 模拟时间延迟

except KeyboardInterrupt:
    print("程序被用户中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    print("正在清理资源...")
    # 停止监听器
    listener.stop()
    listener.join()
    
    # 断开连接
    disconnect()
    print("程序已结束")