import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.spatial.distance import cdist

# ============================================================================
# 系统参数配置
# ============================================================================

# 视图参数
VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080

# 锚点位置（窗口的默认位置）
ANCHOR_POINT = np.array([960, 540])  # 屏幕中心

# 窗口参数
WINDOW_WIDTH = 200
WINDOW_HEIGHT = 200

# 物理参数
K_ATT = 10.0  # 吸引力系数
K_REP = 20.0  # 排斥力系数
D0 = 150.0  # 障碍物影响范围
DT = 1.0  # 时间步长
MAX_V = 5.0  # 最大速度
MASK_UPDATE_RATE = 0.8  # mask更新阻尼系数

# ============================================================================
# 全局状态变量
# ============================================================================

# 窗口状态
window_position = ANCHOR_POINT.copy().astype(float)  # 当前窗口中心位置
window_velocity = np.array([0.0, 0.0])  # 当前窗口速度

# 环境状态
current_mask = np.zeros((VIEW_HEIGHT, VIEW_WIDTH), dtype=bool)  # 当前障碍物mask
obstacles = np.array([]).reshape(0, 2)  # 障碍物坐标列表

# 路径记录
path_history = [window_position.copy()]

# ============================================================================
# 核心功能函数
# ============================================================================

def mask_to_obstacles(mask: np.ndarray, sample_rate: int = 10) -> np.ndarray:
    """
    将mask转换为障碍物坐标列表
    
    Args:
        mask (np.ndarray): 布尔类型mask，True表示障碍物位置
        sample_rate (int): 采样率，每隔多少个像素采样一个障碍物点
    
    Returns:
        np.ndarray: 障碍物坐标数组，shape为(N, 2)
    """
    if not np.any(mask):
        return np.array([]).reshape(0, 2)
    
    # 获取所有障碍物像素的坐标
    obstacle_indices = np.where(mask)
    y_coords = obstacle_indices[0]
    x_coords = obstacle_indices[1]
    
    # 按采样率采样，减少计算量
    if len(x_coords) > 0:
        sample_indices = np.arange(0, len(x_coords), sample_rate)
        sampled_obstacles = np.column_stack((x_coords[sample_indices], y_coords[sample_indices]))
        return sampled_obstacles
    
    return np.array([]).reshape(0, 2)

def update_mask(current_mask: np.ndarray, detection_mask: np.ndarray) -> np.ndarray:
    """
    更新系统mask，使用阻尼更新策略
    
    Args:
        current_mask (np.ndarray): 当前系统mask
        detection_mask (np.ndarray): 检测模型输出的mask
    
    Returns:
        np.ndarray: 更新后的mask
    """
    # 使用阻尼更新策略，避免mask突变
    updated_mask = MASK_UPDATE_RATE * current_mask + (1 - MASK_UPDATE_RATE) * detection_mask
    return updated_mask.astype(bool)

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
    distance = np.linalg.norm(attractive_force)
    
    if distance > 0:
        # 只返回单位向量，不应用系数
        attractive_force = attractive_force / distance
    else:
        attractive_force = np.zeros(2)
    
    return attractive_force

def get_repulsive_force(position: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """
    计算来自障碍物的排斥力（单位向量），考虑窗口位置和锚点的影响范围
    
    Args:
        position (np.ndarray): 窗口中心位置
        obstacles (np.ndarray): 障碍物坐标数组
    
    Returns:
        np.ndarray: 排斥力单位向量
    """
    if len(obstacles) == 0:
        return np.zeros(2)
    
    # 计算窗口中心到障碍物的距离
    window_distances = np.linalg.norm(obstacles - position, axis=1)
    
    # 计算锚点到障碍物的距离
    anchor_distances = np.linalg.norm(obstacles - ANCHOR_POINT, axis=1)
    
    # 取窗口位置和锚点的影响范围并集
    window_influence = window_distances < D0
    anchor_influence = anchor_distances < D0
    combined_influence = window_influence | anchor_influence
    
    if not np.any(combined_influence):
        return np.zeros(2)
    
    # 获取在影响范围内的障碍物
    valid_obstacles = obstacles[combined_influence]
    valid_window_distances = window_distances[combined_influence]
    
    # 计算排斥力（只考虑窗口中心点）
    total_repulsive_force = np.zeros(2)
    
    for i, obs in enumerate(valid_obstacles):
        dist = valid_window_distances[i]
        if dist > 0:
            # 排斥力公式：(1/d - 1/d0) * (1/d^2) * direction
            rep_magnitude = ((1/dist) - (1/D0)) * (1/(dist**2))
            direction = (position - obs) / dist
            repulsive_force = rep_magnitude * direction
            total_repulsive_force += repulsive_force
    
    # 归一化为单位向量
    force_magnitude = np.linalg.norm(total_repulsive_force)
    if force_magnitude > 0:
        total_repulsive_force = total_repulsive_force / force_magnitude
    
    return total_repulsive_force

def check_collision(position: np.ndarray, obstacles: np.ndarray) -> bool:
    """
    检查窗口是否与障碍物发生碰撞
    
    Args:
        position (np.ndarray): 窗口中心位置
        obstacles (np.ndarray): 障碍物坐标数组
    
    Returns:
        bool: 是否发生碰撞
    """
    if len(obstacles) == 0:
        return False
    
    # 计算窗口边界
    window_left = position[0] - WINDOW_WIDTH/2
    window_right = position[0] + WINDOW_WIDTH/2
    window_top = position[1] - WINDOW_HEIGHT/2
    window_bottom = position[1] + WINDOW_HEIGHT/2
    
    # 检查是否有障碍物在窗口内
    for obs in obstacles:
        if (window_left <= obs[0] <= window_right and 
            window_top <= obs[1] <= window_bottom):
            return True
    
    return False

def update_window_position():
    """
    更新窗口位置
    """
    global window_position, window_velocity, obstacles
    
    # 计算合力（应用系数）
    attractive_force = get_attractive_force(window_position, ANCHOR_POINT)
    repulsive_force = get_repulsive_force(window_position, obstacles)
    total_force = K_ATT * attractive_force + K_REP * repulsive_force
    
    # 使用过阻尼模型更新速度
    damping_factor = 0.8  # 阻尼系数
    window_velocity = damping_factor * window_velocity + total_force * DT
    
    # 限制最大速度
    velocity_magnitude = np.linalg.norm(window_velocity)
    if velocity_magnitude > MAX_V:
        window_velocity = window_velocity / velocity_magnitude * MAX_V
    
    # 更新位置
    window_position += window_velocity * DT
    
    # 记录路径
    path_history.append(window_position.copy())

def process_detection_mask(detection_mask: np.ndarray):
    """
    处理检测mask并更新系统状态
    
    Args:
        detection_mask (np.ndarray): 检测模型输出的mask
    """
    global current_mask, obstacles
    
    # 更新系统mask
    current_mask = update_mask(current_mask, detection_mask)
    
    # 转换为障碍物坐标
    obstacles = mask_to_obstacles(current_mask, sample_rate=20)
    
    # 更新窗口位置
    update_window_position()

# ============================================================================
# 可视化功能
# ============================================================================

def setup_visualization():
    """
    设置可视化环境
    
    Returns:
        tuple: (fig, ax) matplotlib图形对象
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, VIEW_WIDTH)
    ax.set_ylim(0, VIEW_HEIGHT)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 图像坐标系y轴向下
    
    # 设置标题和标签
    ax.set_title('AR Window Adaptive Movement System', fontsize=14)
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def visualize_current_state(ax):
    """
    可视化当前状态
    
    Args:
        ax: matplotlib轴对象
    """
    ax.clear()
    ax.set_xlim(0, VIEW_WIDTH)
    ax.set_ylim(0, VIEW_HEIGHT)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # 绘制锚点
    ax.plot(ANCHOR_POINT[0], ANCHOR_POINT[1], 'go', markersize=15, label='Anchor Point')
    
    # 绘制窗口
    window_rect = patches.Rectangle(
        (window_position[0] - WINDOW_WIDTH/2, window_position[1] - WINDOW_HEIGHT/2),
        WINDOW_WIDTH, WINDOW_HEIGHT,
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7, label='Window'
    )
    ax.add_patch(window_rect)
    
    # 绘制窗口中心点
    ax.plot(window_position[0], window_position[1], 'bo', markersize=8)
    
    # 绘制障碍物
    if len(obstacles) > 0:
        ax.scatter(obstacles[:, 0], obstacles[:, 1], c='red', s=10, alpha=0.6, label='Obstacles')
    
    # 绘制路径历史
    if len(path_history) > 1:
        path_array = np.array(path_history)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=1, alpha=0.5, label='Path')
    
    # 绘制影响范围
    influence_circle = plt.Circle((window_position[0], window_position[1]), D0, 
                                 fill=False, color='orange', linestyle='--', alpha=0.5, label='Influence Range')
    ax.add_patch(influence_circle)
    
    # 检查碰撞状态
    collision = check_collision(window_position, obstacles)
    if collision:
        ax.set_title('AR Window Adaptive Movement System - COLLISION DETECTED!', 
                    fontsize=14, color='red')
    else:
        ax.set_title('AR Window Adaptive Movement System', fontsize=14)
    
    ax.legend(loc='upper right')

# ============================================================================
# 测试和演示功能
# ============================================================================

def create_test_mask(frame: int) -> np.ndarray:
    """
    创建测试用的检测mask
    
    Args:
        frame (int): 当前帧数
    
    Returns:
        np.ndarray: 测试mask
    """
    mask = np.zeros((VIEW_HEIGHT, VIEW_WIDTH), dtype=bool)
    
    # 创建移动的障碍物
    t = frame * 0.1
    
    # 障碍物1：在屏幕中心附近移动
    x1 = int(800 + 200 * np.sin(t))
    y1 = int(400 + 150 * np.cos(t))
    mask[max(0, y1-50):min(VIEW_HEIGHT, y1+50), 
         max(0, x1-50):min(VIEW_WIDTH, x1+50)] = True
    
    # 障碍物2：在锚点附近移动
    x2 = int(ANCHOR_POINT[0] + 100 * np.cos(t * 0.5))
    y2 = int(ANCHOR_POINT[1] + 80 * np.sin(t * 0.5))
    mask[max(0, y2-40):min(VIEW_HEIGHT, y2+40), 
         max(0, x2-40):min(VIEW_WIDTH, x2+40)] = True
    
    # 障碍物3：随机出现的小障碍物
    if frame % 100 < 50:
        x3 = int(1200 + 100 * np.sin(t * 2))
        y3 = int(600 + 100 * np.cos(t * 2))
        mask[max(0, y3-30):min(VIEW_HEIGHT, y3+30), 
             max(0, x3-30):min(VIEW_WIDTH, x3+30)] = True
    
    return mask

def run_demo():
    """
    运行演示程序
    """
    fig, ax = setup_visualization()
    
    def animate(frame):
        # 创建测试mask
        test_mask = create_test_mask(frame)
        
        # 处理检测mask
        process_detection_mask(test_mask)
        
        # 可视化当前状态
        visualize_current_state(ax)
        
        return ax,
    
    # 创建动画
    ani = animation.FuncAnimation(fig, animate, frames=500, interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("AR Window Adaptive Movement System")
    print("=" * 50)
    print(f"View size: {VIEW_WIDTH} x {VIEW_HEIGHT}")
    print(f"Window size: {WINDOW_WIDTH} x {WINDOW_HEIGHT}")
    print(f"Anchor point: {ANCHOR_POINT}")
    print(f"Attractive force coefficient: {K_ATT}")
    print(f"Repulsive force coefficient: {K_REP}")
    print(f"Influence range: {D0}")
    print("=" * 50)
    
    # 运行演示
    run_demo()

