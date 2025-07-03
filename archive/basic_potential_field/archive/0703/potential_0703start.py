import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Setting parameters for the potential field
k_att = 10.0  # 吸引力系数
k_rep = 20.0  # 排斥力系数
d0 = 200.0  # 障碍物影响范围
dt = 1  # 时间步长
max_att_force = 50.0
max_rep_force = 50.0  # 最大排斥力限制

# 0702 这里可以改掉，使用更符合mask格式的障碍建立方式，确保后续的格式兼容
# 后续的依赖太多了...以这种格式reshape之后会有更好的处理办法吗？还是对mask先作预处理提取重心点，再计算面积边缘之类的？仔细想想

initial_obstacles = np.array([[526.0, 327.0], [500.0, 700.0]])
obstacle_speeds = np.array([[-0.2, -0.1], [0.5, 0.5]])
obstacles = initial_obstacles.copy()

# robot_position = np.array([0.0, 0.0])
robot_position = np.array([500.0, 700.0])

robot_velocity = np.array([1.0, 1.0])   # 用非零值初始化防止后续归一化时出现除零错误
goal = np.array([500, 700])  # 目标点坐标

# 初始化路径
path_data = [robot_position.copy()]

# 预计算势场和力场
x_limit = (-5, 1920)
y_limit = (-5, 1080)
# x_range = np.linspace(-5, 20, 50)
# y_range = np.linspace(-5, 20, 50)
x_range = np.linspace(x_limit[0], x_limit[1], 100)
y_range = np.linspace(y_limit[0], y_limit[1], 100)
X, Y = np.meshgrid(x_range, y_range)

def attractive_force(position, goal):
    """
    Calculate the attractive force towards the goal
    只输出单位方向向量的吸引力
    """
    attr_force = goal - position
    if not np.all(attr_force == 0):
        attr_force = attr_force / np.linalg.norm(attr_force)
    return attr_force

def repulsive_force(position, obstacles, d0):
    """
    Calculate the repulsive force from obstacles
    只输出单位方向向量的排斥力，向量化实现
    """
    if len(obstacles) == 0:
        return np.zeros(2)
    cur_pos = position.reshape(1, 2)
    obs = np.array(obstacles)
    diff = cur_pos - obs  # (N,2)
    D = np.linalg.norm(diff, axis=1)  # (N,)
    mask = (D < d0) & (D > 0)
    if not np.any(mask):
        return np.zeros(2)
    rep_force = (1 / D[mask] - 1 / d0) * (1 / D[mask]) ** 2
    rep_force = rep_force[:, np.newaxis] * diff[mask]  # (M,2)
    rep_force = np.sum(rep_force, axis=0)
    if not np.all(rep_force == 0):
        rep_force = rep_force / np.linalg.norm(rep_force)
    return rep_force

def total_force(position, goal, obstacles, d0, k_att, k_rep):
    """
    Calculate the total force on the robot
    """
    force_att = attractive_force(position, goal)
    force_rep = repulsive_force(position, obstacles, d0)
    force = k_att * force_att + k_rep * force_rep
    return force


# 封装势场和力场计算为函数
def compute_potential_and_force_field(goal, obstacles, k_att, k_rep, d0):
    # 计算每个网格点到目标的向量差
    diff_goal = np.stack([X - goal[0], Y - goal[1]], axis=-1)  # (H, W, 2)

    # 吸引势场：0.5 * k_att * |r - goal|^2
    att_potential = 0.5 * k_att * np.sum(diff_goal**2, axis=-1)  # (H, W)
    # 吸引力：-k_att * (r - goal)
    force_att = -k_att * diff_goal  # (H, W, 2)

    # 初始化排斥势场
    rep_potential = np.zeros_like(X)  # (H, W)
    # 初始化排斥力
    force_rep = np.zeros(X.shape + (2,))  # (H, W, 2)

    for obs in obstacles:
        # 计算每个网格点到当前障碍物的向量差
        diff_obs = np.stack([X - obs[0], Y - obs[1]], axis=-1)  # (H, W, 2)
        dist = np.linalg.norm(diff_obs, axis=-1)  # (H, W)
        mask = dist < d0  # 只在影响范围内计算排斥

        # 避免除零
        safe_dist = np.where(mask, dist, 1)  # (H, W)

        # 排斥势场：只在mask范围内累加
        rep_potential[mask] += 0.5 * k_rep * ((1/safe_dist[mask]) - (1/d0))**2

        # 排斥力计算（向量化，结果shape为(N,2)）
        # 先计算系数部分，shape为(N,1)
        coeff = k_rep * ((1/safe_dist[mask]) - (1/d0)) * (1/(safe_dist[mask]**2))
        coeff = coeff[:, np.newaxis]  # (N,1)
        # 单位方向向量 (N,2)
        direction = diff_obs[mask] / safe_dist[mask][:, np.newaxis]  # (N,2)
        rep_force = coeff * direction  # (N,2)

        # 限制最大排斥力
        norm_rep = np.linalg.norm(rep_force, axis=-1)  # (N,)
        over = norm_rep > max_rep_force
        if np.any(over):
            rep_force[over] = rep_force[over] / norm_rep[over][:, np.newaxis] * max_rep_force

        # 累加到总排斥力场
        force_rep[mask] += rep_force

    # 总势场和力场
    Z = att_potential + rep_potential  # (H, W)
    U = force_att[..., 0] + force_rep[..., 0]  # (H, W)
    V = force_att[..., 1] + force_rep[..., 1]  # (H, W)
    return Z, U, V

# 初始化势场和力场
Z, U, V = compute_potential_and_force_field(goal, obstacles, k_att, k_rep, d0)

# 设置图形
fig, ax = plt.subplots(figsize=(10, 8))
# ax.set_xlim(-5, 15)
# ax.set_ylim(-5, 15)
ax.set_xlim(x_limit)
ax.set_ylim(y_limit)

# 绘制静态元素
goal_plot, = ax.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')
obstacles_plot, = ax.plot(obstacles[:, 0], obstacles[:, 1], 'ro', markersize=8, label='Obstacles')
path_plot, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
robot_plot, = ax.plot([], [], 'bo', markersize=8, label='Robot')

# 添加窗口矩形（以机器人点为中心，200x100）
window_width = 200
window_height = 100
window_rect = Rectangle((robot_position[0] - window_width/2, robot_position[1] - window_height/2),
                       window_width, window_height, linewidth=2, edgecolor='orange', facecolor='none', label='Window')
ax.add_patch(window_rect)

# 用imshow显示势场
im = ax.imshow(Z, extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]], origin='lower', cmap='viridis', alpha=0.6, aspect='auto')

# 增加势场颜色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Potential Value', fontsize=12)

# 绘制力场箭头
quiver_all = ax.quiver(X, Y, U, V, color='white', alpha=0.5)

# 初始化机器人力向量
robot_force = total_force(robot_position, goal, obstacles, d0, k_att, k_rep)
quiver_robot = ax.quiver([], [], [], [], color='red')

# 添加力数值文本（初始为空）
force_text = ax.text(0.65, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def init():
    """初始化动画"""
    path_plot.set_data([], [])
    robot_plot.set_data([], [])
    return path_plot, robot_plot

def update(frame):
    """更新动画帧"""
    global robot_position, robot_velocity, obstacles, quiver_all, quiver_robot, im, window_rect, force_text
    
    # 计算当前力并更新机器人位置
    force = total_force(robot_position, goal, obstacles, d0, k_att, k_rep)
    robot_velocity += force
    robot_velocity /= np.linalg.norm(robot_velocity)
    robot_position = robot_position + robot_velocity * dt
    path_data.append(robot_position.copy())

    # 更新障碍物位置（动态障碍物）
    moving_factor = 10
    obstacles[:, 0] += obstacle_speeds[:, 0] * np.sin(0.05 * frame) * moving_factor
    obstacles[:, 1] += obstacle_speeds[:, 1] * np.cos(0.05 * frame) * moving_factor

    # 重新计算势场和力场
    Z, U, V = compute_potential_and_force_field(goal, obstacles, k_att, k_rep, d0)
    im.set_data(Z)  # 只更新数据，不重建对象
    quiver_all.set_UVC(U, V)  # 只更新箭头，不重建对象

    # 更新路径和机器人位置
    path = np.array(path_data)
    path_plot.set_data(path[:, 0], path[:, 1])
    robot_plot.set_data([robot_position[0]], [robot_position[1]])
    obstacles_plot.set_data(obstacles[:, 0], obstacles[:, 1])
    
    # 更新机器人力向量箭头并创建新的
    quiver_robot.remove()
    quiver_robot = ax.quiver(robot_position[0], robot_position[1], 
                            force[0], force[1], 
                            color='red', scale=1, width=0.005)

    # 检查是否到达目标
    if np.linalg.norm(robot_position - goal) < 0.5:
        print(f"Goal reached at frame {frame}!")
    
    # 更新窗口矩形位置
    window_rect.set_xy((robot_position[0] - window_width/2, robot_position[1] - window_height/2))
    
    # 更新力数值文本
    force_text.set_text(
        f"attractive_force: [{attractive_force(robot_position, goal)[0]:.2f}, {attractive_force(robot_position, goal)[1]:.2f}]\n"
        f"repulsive_force: [{repulsive_force(robot_position, obstacles, d0)[0]:.2f}, {repulsive_force(robot_position, obstacles, d0)[1]:.2f}]\n"
        f"force:   [{force[0]:.2f}, {force[1]:.2f}]"
    )
    
    return path_plot, robot_plot, obstacles_plot, im, quiver_all, window_rect, force_text

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, 
                             blit=False, interval=50, repeat=False)

# 设置图例和标签
ax.legend(loc='upper left')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Dynamic Artificial Potential Field Path Planning')
ax.grid(True, alpha=0.3)

# 显示动画
plt.tight_layout()
plt.show()

# 可选：保存动画为gif
# ani.save('dynamic_potential_field.gif', writer='pillow', fps=30)