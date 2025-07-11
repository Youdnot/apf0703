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

# # 可视化的坐标轴好像有点问题
# fig, ax = plt.subplots(figsize=(10, 8))

# ax.set_xlim(-100, view_width + 100)
# ax.set_ylim(-100, view_height + 100)

# ax.set_title('Dynamic Artificial Potential Field')

# # 绘制静态元素
# anchor_plot, = ax.plot(anchor_point[0], anchor_point[1], 'go', markersize=10, label='Anchor Point')
# # obstacles_plot, = ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro', markersize=8, label='Obstacles')

# window_plot, = ax.plot(cur_pos[0], cur_pos[1], 'bo', markersize=8, label='Window Center')

# # 添加缺失的path_plot定义
# path_plot, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
# window_plot, = ax.plot([], [], 'bo', markersize=8, label='Window Center')

# # 绘制窗口
# window_rect = patches.Rectangle(
#     (cur_pos[0] - window_width/2, cur_pos[1] - window_height/2),
#     window_width, window_height,
#     linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7, label='Window'
# )
# ax.add_patch(window_rect)

# # 绘制锚点影响范围
# window_influence = patches.Circle((cur_pos[0], cur_pos[1]), radius=d0, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.05, label='Window Influence Range')
# ax.add_patch(window_influence)

# # 绘制锚点影响范围
# anchor_influence = patches.Circle((anchor_point[0], anchor_point[1]), radius=d0, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.05, label='Anchor Influence Range')
# ax.add_patch(anchor_influence)

# test_plot, = ax.plot([0], [0], 'go', markersize=8, label='Axis Test')

# # 使用imshow绘制障碍物
# obstacle_plot = ax.imshow(obstacle_mask.T, origin='lower', 
#                           extent=(0, view_width, 0, view_height),
#                           cmap='Reds', alpha=0.5)

# # 绘制吸引力箭头测试
# attractive_force_vis = attractive_force * 9
# repulsive_force_vis = repulsive_force * 9
# attractive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], attractive_force_vis[0], attractive_force_vis[1],
#                          width=5, color='blue')
# repulsive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], repulsive_force_vis[0], repulsive_force_vis[1],
#                          width=5, color='red')

# # 添加总力箭头
# total_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], total_force[0], total_force[1],
#                          width=5, color='green')

# def init():
#     """初始化动画"""
#     path_plot.set_data([], [])
#     window_plot.set_data([], [])
#     return path_plot, window_plot

# def update_plot(frame):
#     """更新动画帧"""
#     global cur_pos, cur_vel, converted_pos, obstacle_mask, view_width, view_height, d0, k_att, k_rep, max_v, path_plot, window_plot, path_data
    
#     # 更新位置和速度
#     force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, anchor_point, obstacle_mask, view_width, view_height, d0, k_att, k_rep, damping_factor, max_v, dt, path_data)

#     # 可视化更新部份
#     path = np.array(path_data)
#     path_plot.set_data(path[:, 0], path[:, 1])
#     window_plot.set_data([cur_pos[0]], [cur_pos[1]])

#     # 更新窗口及其影响范围
#     window_rect.set_xy((cur_pos[0] - window_width/2, cur_pos[1] - window_height/2))
#     window_influence.center = cur_pos[0], cur_pos[1]

#     # 更新力向量箭头 - 修复变量作用域问题
#     global attractive_force_arraw, repulsive_force_arraw, total_force_arraw

#     attractive_force = get_attractive_force(cur_pos, anchor_point) * k_att * 9
#     repulsive_force = get_repulsive_force(cur_pos, anchor_point, obstacle_mask, view_width, view_height, d0) * k_rep * 9
    
#     # 移除旧的箭头
#     if 'attractive_force_arraw' in globals():
#         attractive_force_arraw.remove()
#     if 'repulsive_force_arraw' in globals():
#         repulsive_force_arraw.remove()
#     if 'total_force_arraw' in globals():
#         total_force_arraw.remove()

#     attractive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], attractive_force[0], attractive_force[1],
#                          width=5, color='blue')

#     repulsive_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], repulsive_force[0], repulsive_force[1],
#                          width=5, color='red')
                         
#     total_force_arraw = plt.arrow(cur_pos[0], cur_pos[1], 9 * force[0], 9 * force[1],
#                          width=5, color='green')

#     return path_plot, window_plot, attractive_force_arraw, repulsive_force_arraw, total_force_arraw


# # 创建动画，需要重新调整可视化并封装图像的init和参数的update以传入，后续再调整
# ani = animation.FuncAnimation(fig, update_plot, frames=2000, init_func=init, 
#                              blit=False, interval=50, repeat=False)

# # 外围图例
# ax.legend(loc='upper left')
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')

# plt.tight_layout()
# plt.show()

# UI control
from utils.control_unity_ui import *

# Initialize connection and create element
element_key = initialize_connection()

import time

for i in range(2000):
    # 更新位置和速度
    force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, anchor_point, obstacle_mask, view_width, view_height, d0, k_att, k_rep, damping_factor, max_v, dt, path_data)
    update_position(converted_pos)
    time.sleep(0.1)  # 模拟时间延迟


listener = keyboard.Listener(on_press=on_press)
listener.start()

stop_event.wait()

# Clean up
disconnect()
listener.join()