import numpy as np

def get_attractive_force(position: np.ndarray, anchor: np.ndarray) -> np.ndarray:
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
    
    if modulus > 0:
        # 返回单位向量
        attractive_force = attractive_force / modulus
    else:
        attractive_force = np.zeros(2)
    
    return attractive_force

def get_repulsive_force(position: np.ndarray, anchor: np.ndarray, 
                       obstacle_mask: np.ndarray, view_width: int, 
                       view_height: int, d0: float) -> np.ndarray:
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
    # 生成坐标网格
    x_coords, y_coords = np.mgrid[0:view_width, 0:view_height]
    
    # 计算到当前位置和锚点的距离矩阵
    window_distances = np.sqrt((x_coords - position[0])**2 + (y_coords - position[1])**2)
    anchor_distances = np.sqrt((x_coords - anchor[0])**2 + (y_coords - anchor[1])**2)
    
    # 避免除零错误，设置最小距离
    window_distances = np.maximum(window_distances, 1e-6)
    anchor_distances = np.maximum(anchor_distances, 1e-6)
    
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
    window_distances_x[window_distances_x == 0]= 1e-6
    window_distances_y[window_distances_y == 0]= 1e-6
    
    # 计算各方向的排斥力分量
    force_x = repulsive_coefficient * window_distances_x
    force_y = repulsive_coefficient * window_distances_y
    
    # 汇总排斥力
    total_force_x = np.sum(force_x)
    total_force_y = np.sum(force_y)
    
    repulsive_force = np.array([total_force_x, total_force_y])
    modulus = np.linalg.norm(repulsive_force)
    
    # 返回单位向量
    if modulus > 0:
        repulsive_force = repulsive_force / modulus
    else:
        repulsive_force = np.zeros(2)
    
    return repulsive_force

def get_total_force(position: np.ndarray, anchor: np.ndarray, obstacle_mask: np.ndarray, view_width: int, view_height: int, d0: float, k_att: float, k_rep: float) -> np.ndarray:
    attractive_force = get_attractive_force(position, anchor)
    repulsive_force = get_repulsive_force(position, anchor, obstacle_mask, view_width, view_height, d0)
    total_force = k_att * attractive_force + k_rep * repulsive_force
    return total_force