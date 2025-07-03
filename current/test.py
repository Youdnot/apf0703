import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

# ==================== 参数设置 ====================
# 基础视图参数
view_width = 1920
view_height = 1080

# 锚点位置（窗口的目标位置）
anchor_point = np.array([500, 700])

# 窗口参数
window_width = 200
window_height = 200

# 当前位置和速度
cur_pos = anchor_point.copy()
cur_vel = np.array([0, 0])

# 最大速度限制
max_v = 10

# 力场参数
k_att = zeta = 10.0  # 吸引力系数
k_rep = eta = 10.0   # 排斥力系数
d0 = 100             # 障碍物影响范围

# ==================== 创建测试障碍物mask ====================
# 创建一个不规则的障碍物mask用于测试
obstacle_mask = np.zeros((view_width, view_height), dtype=bool)

# 创建多个不规则形状的障碍物
# 障碍物1：矩形
obstacle_mask[500:600, 600:700] = True

# 障碍物2：圆形区域
y, x = np.ogrid[:view_height, :view_width]
center_y, center_x = 300, 400
radius = 80
circle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
obstacle_mask |= circle_mask

# 障碍物3：不规则多边形
polygon_points = np.array([[800, 300], [900, 350], [850, 450], [750, 400]])
from matplotlib.path import Path
polygon_path = Path(polygon_points)
y_coords, x_coords = np.mgrid[0:view_height, 0:view_width]
points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
polygon_mask = polygon_path.contains_points(points).reshape(view_height, view_width)
obstacle_mask |= polygon_mask

# ==================== 高效的排斥力计算 ====================
# 创建坐标网格 - 用于计算每个像素到当前位置的距离
y_coords, x_coords = np.mgrid[0:view_height, 0:view_width]

# 计算当前位置到每个像素的距离矩阵
distances = np.sqrt((x_coords - cur_pos[0])**2 + (y_coords - cur_pos[1])**2)

# 避免除零错误，设置最小距离阈值
distances = np.maximum(distances, 1e-6)

# 计算排斥力强度：使用改进的势场函数
# 当距离小于d0时产生排斥力，使用平滑的过渡函数
influence_mask = distances < d0
force_magnitude = np.zeros_like(distances)

# 在影响范围内计算排斥力强度
# 使用改进的势场函数：F = k_rep * (1/d - 1/d0) * (1/d)^2 * exp(-d/(d0*0.3))
force_magnitude[influence_mask] = (
    k_rep * 
    (1/distances[influence_mask] - 1/d0) * 
    (1/distances[influence_mask])**2 * 
    np.exp(-distances[influence_mask] / (d0 * 0.3))
)

# 计算力的方向向量（从障碍物指向当前位置）
force_direction_x = (cur_pos[0] - x_coords) / distances
force_direction_y = (cur_pos[1] - y_coords) / distances

# 只在障碍物位置应用力，计算总的排斥力
# 使用向量化操作，避免循环
force_x = np.sum(force_magnitude * force_direction_x * obstacle_mask)
force_y = np.sum(force_magnitude * force_direction_y * obstacle_mask)

repulsive_force = np.array([force_x, force_y])

# 归一化排斥力向量（如果力不为零）
force_magnitude_total = np.linalg.norm(repulsive_force)
if force_magnitude_total > 1e-6:
    repulsive_force = repulsive_force / force_magnitude_total

# ==================== 吸引力计算 ====================
# 计算对锚点的吸引力
attractive_force = anchor_point - cur_pos
attractive_distance = np.linalg.norm(attractive_force)
if attractive_distance > 0:
    attractive_force = attractive_force / attractive_distance

# ==================== 合力计算 ====================
# 计算总的合力
total_force = k_att * attractive_force + k_rep * repulsive_force

# ==================== 调试信息输出 ====================
# 创建详细的数据字典用于调试和可读性
debug_data = {
    'current_position': cur_pos,
    'anchor_position': anchor_point,
    'current_velocity': cur_vel,
    'influence_range': d0,
    'attractive_coefficient': k_att,
    'repulsive_coefficient': k_rep,
    'obstacle_mask_shape': obstacle_mask.shape,
    'obstacle_pixel_count': np.sum(obstacle_mask),
    'min_distance_to_obstacle': np.min(distances[obstacle_mask]),
    'in_influence_range': np.min(distances[obstacle_mask]) < d0,
    'attractive_force': attractive_force,
    'repulsive_force': repulsive_force,
    'total_force': total_force,
    'force_magnitude_total': force_magnitude_total
}

# 输出调试信息
print("=== 人工势场排斥力计算调试信息 ===")
print(f"当前位置: {debug_data['current_position']}")
print(f"锚点位置: {debug_data['anchor_position']}")
print(f"当前速度: {debug_data['current_velocity']}")
print(f"障碍物影响范围: {debug_data['influence_range']}")
print(f"吸引力系数: {debug_data['attractive_coefficient']}")
print(f"排斥力系数: {debug_data['repulsive_coefficient']}")
print(f"障碍物mask形状: {debug_data['obstacle_mask_shape']}")
print(f"障碍物像素数量: {debug_data['obstacle_pixel_count']}")
print(f"到最近障碍物的距离: {debug_data['min_distance_to_obstacle']:.2f}")
print(f"是否在影响范围内: {debug_data['in_influence_range']}")
print(f"吸引力向量: {debug_data['attractive_force']}")
print(f"排斥力向量: {debug_data['repulsive_force']}")
print(f"合力向量: {debug_data['total_force']}")
print(f"排斥力大小: {debug_data['force_magnitude_total']:.6f}")

# ==================== 性能分析 ====================
# 计算性能指标
import time

# 测试计算时间
start_time = time.time()
for _ in range(100):
    # 重新计算排斥力（模拟实时计算）
    distances = np.sqrt((x_coords - cur_pos[0])**2 + (y_coords - cur_pos[1])**2)
    distances = np.maximum(distances, 1e-6)
    influence_mask = distances < d0
    force_magnitude = np.zeros_like(distances)
    force_magnitude[influence_mask] = (
        k_rep * 
        (1/distances[influence_mask] - 1/d0) * 
        (1/distances[influence_mask])**2 * 
        np.exp(-distances[influence_mask] / (d0 * 0.3))
    )
    force_direction_x = (cur_pos[0] - x_coords) / distances
    force_direction_y = (cur_pos[1] - y_coords) / distances
    force_x = np.sum(force_magnitude * force_direction_x * obstacle_mask)
    force_y = np.sum(force_magnitude * force_direction_y * obstacle_mask)
    repulsive_force = np.array([force_x, force_y])
    force_magnitude_total = np.linalg.norm(repulsive_force)
    if force_magnitude_total > 1e-6:
        repulsive_force = repulsive_force / force_magnitude_total

computation_time = time.time() - start_time
avg_time_per_calculation = computation_time / 100

performance_data = {
    'total_calculation_time': computation_time,
    'average_time_per_calculation': avg_time_per_calculation,
    'calculations_per_second': 1.0 / avg_time_per_calculation,
    'mask_size': obstacle_mask.size,
    'memory_usage_mb': obstacle_mask.nbytes / (1024 * 1024)
}

print("\n=== 性能分析 ===")
print(f"100次计算总时间: {performance_data['total_calculation_time']:.4f}秒")
print(f"单次计算平均时间: {performance_data['average_time_per_calculation']:.6f}秒")
print(f"计算频率: {performance_data['calculations_per_second']:.2f}次/秒")
print(f"mask大小: {performance_data['mask_size']}像素")
print(f"内存使用: {performance_data['memory_usage_mb']:.2f}MB")

# ==================== 可视化验证 ====================
# 创建可视化图表验证计算结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 左图：显示障碍物mask和当前位置
ax1.imshow(obstacle_mask.T, origin='lower', cmap='Reds', alpha=0.7)
ax1.plot(cur_pos[0], cur_pos[1], 'bo', markersize=10, label='当前位置')
ax1.plot(anchor_point[0], anchor_point[1], 'go', markersize=10, label='锚点')
ax1.set_title('障碍物分布和位置')
ax1.set_xlabel('X坐标')
ax1.set_ylabel('Y坐标')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：显示力向量
ax2.quiver(cur_pos[0], cur_pos[1], 
           attractive_force[0] * 50, attractive_force[1] * 50, 
           color='green', alpha=0.7, label='吸引力', scale=1)
ax2.quiver(cur_pos[0], cur_pos[1], 
           repulsive_force[0] * 50, repulsive_force[1] * 50, 
           color='red', alpha=0.7, label='排斥力', scale=1)
ax2.quiver(cur_pos[0], cur_pos[1], 
           total_force[0] * 30, total_force[1] * 30, 
           color='blue', alpha=0.9, label='合力', scale=1)
ax2.plot(cur_pos[0], cur_pos[1], 'ko', markersize=8)
ax2.plot(anchor_point[0], anchor_point[1], 'go', markersize=10)
ax2.set_title('力向量可视化')
ax2.set_xlabel('X坐标')
ax2.set_ylabel('Y坐标')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(cur_pos[0] - 200, cur_pos[0] + 200)
ax2.set_ylim(cur_pos[1] - 200, cur_pos[1] + 200)

plt.tight_layout()
plt.show()

# ==================== 最终结果输出 ====================
print("\n=== 最终计算结果 ===")
print(f"排斥力计算完成，向量: {repulsive_force}")
print(f"吸引力向量: {attractive_force}")
print(f"合力向量: {total_force}")
print("计算过程使用纯numpy向量化操作，无需提取障碍物坐标")
print("适用于实时AR环境中的窗口自适应移动")