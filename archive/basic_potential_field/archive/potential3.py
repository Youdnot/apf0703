import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setting parameters for the potential field
k_att = 4.0  # 吸引力系数
k_rep = 10.0  # 排斥力系数
d0 = 4.0  # 障碍物影响范围
dt = 0.01  # 时间步长
max_rep_force = 10.0  # 最大排斥力限制
goal = np.array([10, 10])  # 目标点坐标
initial_obstacles = np.array([[3, 3.5], [6, 7], [9, 9], [8, 4], [5, 5]])
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])

robot_position = np.array([0.0, 0.0])
# robot_position = np.array([10.0, 10.0])
path_data = [robot_position.copy()]
obstacles = initial_obstacles.copy()

# 预计算势场和力场
x_limit = (-5, 20)
y_limit = (-5, 20)
# x_range = np.linspace(-5, 20, 50)
# y_range = np.linspace(-5, 20, 50)
x_range = np.linspace(x_limit[0], x_limit[1], 50)
y_range = np.linspace(y_limit[0], y_limit[1], 50)
X, Y = np.meshgrid(x_range, y_range)


def attractive_force(position, goal, k_att):
    """
    Calculate the attractive force towards the goal
    """
    return -k_att * (position - goal)

def repulsive_force(position, obstacles, d0, k_rep):
    """
    Calculate the repulsive force from obstacles
    """
    force = np.zeros(2)
    for obs in obstacles:
        diff = position - obs
        dist = np.linalg.norm(diff)
        if dist < d0:
            repulsive_force = k_rep * ((1/dist) - (1/d0)) * (1/(dist**2)) * (diff/dist)
            if np.linalg.norm(repulsive_force) > max_rep_force:
                repulsive_force = repulsive_force / np.linalg.norm(repulsive_force) * max_rep_force
            force += repulsive_force
    return force

def total_force(position, goal, obstacles, d0, k_att, k_rep):
    """
    Calculate the total force on the robot
    """
    force_att = attractive_force(position, goal, k_att)
    force_rep = repulsive_force(position, obstacles, d0, k_rep)
    return force_att + force_rep


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

# 用imshow显示势场
im = ax.imshow(Z, extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]], origin='lower', cmap='viridis', alpha=0.6, aspect='auto')

# 绘制力场箭头
quiver_all = ax.quiver(X, Y, U, V, color='white', alpha=0.5)

# 初始化机器人力向量
robot_force = total_force(robot_position, goal, obstacles, d0, k_att, k_rep)
quiver_robot = ax.quiver([], [], [], [], color='red')

def init():
    """初始化动画"""
    path_plot.set_data([], [])
    robot_plot.set_data([], [])
    return path_plot, robot_plot

def update(frame):
    """更新动画帧"""
    global robot_position, obstacles, quiver_all, quiver_robot, im
    
    # 计算当前力并更新机器人位置
    force = total_force(robot_position, goal, obstacles, d0, k_att, k_rep)
    robot_position = robot_position + force * dt
    path_data.append(robot_position.copy())

    # 更新障碍物位置（动态障碍物）
    obstacles[:, 0] += obstacle_speeds[:, 0] * np.sin(0.05 * frame)
    obstacles[:, 1] += obstacle_speeds[:, 1] * np.cos(0.05 * frame)
    
    # 边界检查并反弹
    for obs_idx, obs in enumerate(obstacles):
        if obs[0] < -5 or obs[0] > 15:
            obstacles[obs_idx, 0] = np.clip(obs[0], -5, 15)
            obstacle_speeds[obs_idx, 0] *= -1
        if obs[1] < -5 or obs[1] > 15:
            obstacles[obs_idx, 1] = np.clip(obs[1], -5, 15)
            obstacle_speeds[obs_idx, 1] *= -1

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
    if np.linalg.norm(robot_position - goal) < 0.3:
        print(f"Goal reached at frame {frame}!")
    
    return path_plot, robot_plot, obstacles_plot, im, quiver_all

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