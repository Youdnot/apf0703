# 从左下角为原点的1080p坐标系
# 转换为
# 以中心点为原点的Unity Camera Space
# 并完成scale的缩放 pixel to factor

import numpy as np
import time

config_1080 = {
    'scaling_factor': 0.1/338,
    'center_x': 940,
    'center_y': 576,
}

config_720 = {
    'scaling_factor': 0.1/226,
    'center_x': 618,
    'center_y': 329,
}

def convert_coordinates(position, scaling_factor, center_x, center_y, depth=0.5):
    """
    Convert position from 1080p camera space to Unity camera space.
    """
    view_position = position.copy()
    # Adjust to center of camera space
    view_position -= np.array([center_x, center_y])
    # Adjust amplitude from image pixel space to Unity camera space
    view_position = view_position.astype(np.float32)
    view_position *= scaling_factor
    # Add depth to the z-axis
    view_position = np.append(view_position, depth)
    return view_position

#------------------------------------------------------------------------------

def get_attractive_force(position: np.ndarray, anchor: np.ndarray):
    """
    计算对锚点的吸引力（单位向量）
    
    Args:
        position (np.ndarray): 当前位置
        anchor (np.ndarray): 锚点位置
    
    Returns:
        np.ndarray: 吸引力单位向量
    """
    attractive_force = anchor - position
    modulus = np.linalg.norm(attractive_force)

    # limit the force
    limit_value = 40
    if modulus > limit_value:
        attractive_force = attractive_force*(limit_value/modulus)
    
    return attractive_force

# 生成坐标网格
# x_coords, y_coords = np.mgrid[0:1920, 0:1080]
x_coords, y_coords = np.mgrid[0:1280, 0:720]

def get_repulsive_force(position: np.ndarray, anchor: np.ndarray, 
                       obstacle_mask: np.ndarray, d0: float, x_coords, y_coords):
    """
    计算排斥力（单位向量）
    
    基于人工势场法计算排斥力，考虑障碍物对机器人的排斥作用。
    排斥力计算公式：rep_force = (1/D - 1/d0) * (1/D)^2 * (position - obstacles)
    
    Args:
        position (np.ndarray): 当前位置 [x, y]
        anchor (np.ndarray): 锚点位置 [x, y]
        obstacle_mask (np.ndarray): 障碍物掩码矩阵，True表示障碍物位置
        view_width (int): 视图宽度
        view_height (int): 视图高度
        d0 (float): 障碍物影响范围
    
    Returns:
        np.ndarray: 排斥力单位向量 [x, y]
    """
    
    if obstacle_mask.any():
        # 计算到当前位置和锚点的距离矩阵
        window_distances = np.sqrt((x_coords - position[0])**2 + (y_coords - position[1])**2)
        anchor_distances = np.sqrt((x_coords - anchor[0])**2 + (y_coords - anchor[1])**2)
        
        # 避免除零错误，设置最小距离
        window_distances = np.maximum(window_distances, 1)
        anchor_distances = np.maximum(anchor_distances, 1)
        
        # 计算影响范围掩码
        window_influence_mask = window_distances <= d0
        anchor_influence_mask = anchor_distances <= d0
        # 按位或进行合并
        influence_mask = window_influence_mask | anchor_influence_mask
        # 按位与进行筛选真正起作用的obstacles
        final_mask = obstacle_mask & influence_mask
        
        # 计算排斥力系数：repulsive_coefficient = (1/D - 1/d0) * (1/D)^2
        inv_distances = 1.0 / window_distances
        inv_d0 = 1.0 / d0
        repulsive_coefficient = (inv_distances - inv_d0) * (inv_distances ** 2)
        
        # 只在有效障碍物位置应用排斥力
        repulsive_coefficient = np.where(final_mask, repulsive_coefficient, 0)
        
        # 计算每个网格到窗口点的距离矩阵
        # 这里减完有1单位的误差？索引从0开始
        window_distances_x = position[0] - x_coords
        window_distances_y = position[1] - y_coords

        # 替换0值，避免除零
        # window_distances_x[window_distances_x == 0]= 1e-6
        # window_distances_y[window_distances_y == 0]= 1e-6
        
        # 计算各方向的排斥力分量
        force_x = repulsive_coefficient * window_distances_x
        force_y = repulsive_coefficient * window_distances_y
        
        # 汇总排斥力
        total_force_x = np.sum(force_x)
        total_force_y = np.sum(force_y)
        
        repulsive_force = np.array([total_force_x, total_force_y])

        # 处理停滞 - 如果周围障碍物密度过高，添加小扰动
        density_threshold = 0.9  # 密度阈值
        small_magnitude = 0.1  # 可调整扰动幅度

        obstacle_density = np.sum(final_mask) / np.sum(influence_mask) if np.sum(influence_mask) > 0 else 0
        
        if obstacle_density > density_threshold:
            # 计算吸引力的单位向量作为定向扰动
            attractive_force = get_attractive_force(position, anchor)
            # 添加定向扰动
            # 这里取负号是因为要反向扰动，因为吸引力是向锚点方向的
            perturbation = -attractive_force/np.linalg.norm(attractive_force)*small_magnitude
            
            # 限制斥力幅度，避免过大
            repulsive_force *= 0.5
            
            repulsive_force += perturbation
        
        modulus = np.linalg.norm(repulsive_force)

        # limit the force
        limit_value = 100
        if modulus > limit_value:
            repulsive_force = repulsive_force*(limit_value/modulus)

    else:
        repulsive_force = np.zeros(2)
    
    return repulsive_force

def get_total_force(position: np.ndarray, anchor: np.ndarray, obstacle_mask: np.ndarray, view_width: int, view_height: int, d0: float, k_att: float, k_rep: float):
    attractive_force = get_attractive_force(position, anchor)
    repulsive_force = get_repulsive_force(position, anchor, obstacle_mask, view_width, view_height, d0)
    # print(f"Attractive Force: {attractive_force}, Repulsive Force: {repulsive_force}")
    total_force = k_att * attractive_force + k_rep * repulsive_force
    return total_force

def update_position_and_velocity(cur_pos: np.ndarray, cur_vel: np.ndarray, 
    anchor_point: np.ndarray, obstacle_mask: np.ndarray, 
    view_width: int, view_height: int, d0: float, 
    k_att: float, k_rep: float, damping_factor: float, 
    max_v: float, dt: float):  
    """
    更新机器人位置和速度
    
    基于人工势场法计算合力，并更新机器人的位置和速度。
    
    Args:
        current_position (np.ndarray): 当前位置 [x, y]
        current_velocity (np.ndarray): 当前速度 [vx, vy]
        anchor_point (np.ndarray): 锚点位置 [x, y]
        obstacle_mask (np.ndarray): 障碍物掩码矩阵
        view_width (int): 视图宽度
        view_height (int): 视图高度
        d0 (float): 障碍物影响范围
        k_att (float): 吸引力系数
        k_rep (float): 排斥力系数
        damping_factor (float): 阻尼系数
        max_v (float): 最大速度
        dt (float): 时间步长
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (新位置, 新速度, 合力)
    """
    # 计算当前力
    force = get_total_force(cur_pos, anchor_point, obstacle_mask, view_width, view_height, d0, k_att, k_rep)
    # print(f"Force: {force}, Force Norm: {np.linalg.norm(force)}")
    
    cur_vel = damping_factor * force + (1 - damping_factor) * cur_vel
    cur_pos = cur_pos + cur_vel*max_v*dt

    # test convertion
    converted_pos = convert_coordinates(cur_pos)

    # print(f"Current Position: {cur_pos}, Current Velocity: {cur_vel}, Converted Position: {converted_pos}")

    return force, cur_pos, cur_vel, converted_pos