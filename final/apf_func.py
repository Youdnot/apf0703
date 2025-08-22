import numpy as np
from multiprocessing import Queue
import rerun as rr


# Cache for coordinate grids (global to avoid repeated computation)
_X_COORDS, _Y_COORDS = np.mgrid[0:1280, 0:720]


def _convert_coordinates(position, scaling_factor=0.1/226, center_x=618, center_y=329, depth=0.5):
    """
    Convert position from camera space to Unity camera space.
    从左下角为原点的投影平面坐标系转换为
    以中心点为原点的Unity Camera Space
    并完成scale的缩放 pixel to factor
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


def get_attractive_force(anchor, position, att_limit=40):
    """计算吸引力"""
    attractive_force = anchor - position
    modulus = np.linalg.norm(attractive_force)

    # limit the force
    if modulus > att_limit:
        attractive_force = attractive_force * (att_limit / modulus)
    
    return attractive_force


def get_repulsive_force(position, anchor, obstacle_mask, 
                       d0=400.0, rep_limit=100, rep_density_threshold=0.9, 
                       rep_small_magnitude=0.1, att_limit=40):
    """计算排斥力"""
    if obstacle_mask.any():
        # 计算到当前位置和锚点的距离矩阵
        window_distances = np.sqrt((_X_COORDS - position[0])**2 + (_Y_COORDS - position[1])**2)
        anchor_distances = np.sqrt((_X_COORDS - anchor[0])**2 + (_Y_COORDS - anchor[1])**2)
        
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
        window_distances_x = position[0] - _X_COORDS
        window_distances_y = position[1] - _Y_COORDS

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

        obstacle_density = np.sum(final_mask) / np.sum(influence_mask) if np.sum(influence_mask) > 0 else 0
        
        if obstacle_density > rep_density_threshold:
            # 计算吸引力的单位向量作为定向扰动
            attractive_force = get_attractive_force(anchor, position, att_limit)
            # 添加定向扰动
            # 这里取负号是因为要反向扰动，因为吸引力是向锚点方向的
            perturbation = -attractive_force / np.linalg.norm(attractive_force) * rep_small_magnitude
            
            # 限制斥力幅度，避免过大
            repulsive_force *= 0.5
            
            repulsive_force += perturbation
        
        modulus = np.linalg.norm(repulsive_force)

        # limit the force
        if modulus > rep_limit:
            repulsive_force = repulsive_force * (rep_limit / modulus)

    else:
        repulsive_force = np.zeros(2)
    
    return repulsive_force


def get_total_force(anchor, position, obstacle_mask, k_att=0.4, k_rep=30.0, 
                   d0=400.0, att_limit=40, rep_limit=100, rep_density_threshold=0.9, 
                   rep_small_magnitude=0.1):
    """计算总力"""
    attractive_force = get_attractive_force(anchor, position, att_limit)
    repulsive_force = get_repulsive_force(position, anchor, obstacle_mask, 
                                        d0, rep_limit, rep_density_threshold, 
                                        rep_small_magnitude, att_limit)
    # print(f"Attractive Force: {attractive_force}, Repulsive Force: {repulsive_force}")
    total_force = k_att * attractive_force + k_rep * repulsive_force
    return total_force


def apf_calculate(anchor, position, obstacle_mask, velocity=0,
                 # Physics parameters
                 k_att=0.4, k_rep=50.0, damping_factor=1.0, d0=600.0, 
                 max_v=15.0, dt=0.1, att_limit=40, rep_limit=120,
                 rep_density_threshold=0.9, rep_small_magnitude=0.1,
                 # Coordinate conversion parameters  
                 scaling_factor=0.1/226, center_x=618, center_y=329, depth=0.5):
    """
    人工势场计算函数，用于在进程中调用
    
    Args:
        anchor: 目标锚点位置
        position: 当前位置
        obstacle_mask: 障碍物掩码
        velocity: 当前速度 (默认为0)
        
        Physics parameters:
        k_att: 吸引力系数 (默认0.4)
        k_rep: 排斥力系数 (默认30.0)
        damping_factor: 阻尼系数 (默认1.0)
        d0: 障碍物影响范围 (默认400.0)
        max_v: 最大速度 (默认15.0)
        dt: 时间步长 (默认0.1)
        att_limit: 吸引力限制 (默认40)
        rep_limit: 排斥力限制 (默认100)
        rep_density_threshold: 密度阈值 (默认0.9)
        rep_small_magnitude: 扰动幅度 (默认0.1)
        
        Coordinate conversion parameters:
        scaling_factor: 缩放因子 (默认0.1/226 for 720p)
        center_x: 中心x坐标 (默认618 for 720p)
        center_y: 中心y坐标 (默认329 for 720p)
        depth: 深度值 (默认0.5)
    
    Returns:
        tuple: (updated_position, updated_velocity, converted_position)
    """
    # 计算当前力
    force = get_total_force(anchor, position, obstacle_mask, k_att, k_rep, 
                           d0, att_limit, rep_limit, rep_density_threshold, 
                           rep_small_magnitude)
    
    # 更新速度和位置
    new_velocity = damping_factor * force + (1 - damping_factor) * velocity
    new_position = position + (new_velocity * max_v * dt)

    # 转换坐标
    converted_pos = _convert_coordinates(new_position, scaling_factor, center_x, center_y, depth)

    return new_position, new_velocity, converted_pos