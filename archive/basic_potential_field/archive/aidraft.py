# 这一版在4的基础上重构了参数的设置，在数据读写上存在问题，无法正常运行

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setting parameters for the potential field
k_att = 4.0  # 吸引力系数
k_rep = 10.0  # 排斥力系数
d0 = 4.0  # 障碍物影响范围
dt = 0.01  # 时间步长
max_rep_force = 10.0  # 最大排斥力限制

initial_obstacles = np.array([[3, 3.5], [6, 7], [9, 9], [8, 4], [5, 5]])
obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.2, 0.1]])
obstacles = initial_obstacles.copy()

robot_position = np.array([0.0, 0.0])
# robot_position = np.array([10.0, 10.0])
path_data = [robot_position.copy()]
goal = np.array([10, 10])  # 目标点坐标

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

# ================= 可视化与动画模块 =================
class Visualize:
    def __init__(self, Z, U, V, X, Y, robot_position, goal, obstacles, path_data, x_limit, y_limit, show=True):
        self.Z = Z
        self.U = U
        self.V = V
        self.X = X
        self.Y = Y
        self.robot_position = robot_position
        self.goal = goal
        self.obstacles = obstacles
        self.path_data = path_data
        self.x_limit = x_limit
        self.y_limit = y_limit
        self.show = show
        self.fig = None
        self.ax = None
        self.im = None
        self.quiver_all = None
        self.goal_plot = None
        self.obstacles_plot = None
        self.path_plot = None
        self.robot_plot = None
        self.quiver_robot = None
        self.ani = None

    def setup(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(self.x_limit)
        self.ax.set_ylim(self.y_limit)
        self.goal_plot, = self.ax.plot(self.goal[0], self.goal[1], 'go', markersize=10, label='Goal')
        self.obstacles_plot, = self.ax.plot(self.obstacles[:, 0], self.obstacles[:, 1], 'ro', markersize=8, label='Obstacles')
        self.path_plot, = self.ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
        self.robot_plot, = self.ax.plot([], [], 'bo', markersize=8, label='Robot')
        self.im = self.ax.imshow(self.Z, extent=[self.x_limit[0], self.x_limit[1], self.y_limit[0], self.y_limit[1]], origin='lower', cmap='viridis', alpha=0.6, aspect='auto')
        cbar = plt.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label('Potential Value', fontsize=12)
        self.quiver_all = self.ax.quiver(self.X, self.Y, self.U, self.V, color='white', alpha=0.5)
        self.quiver_robot = self.ax.quiver([], [], [], [], color='red')
        self.ax.legend(loc='upper left')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Dynamic Artificial Potential Field Path Planning')
        self.ax.grid(True, alpha=0.3)
        plt.tight_layout()

    def init_anim(self):
        self.path_plot.set_data([], [])
        self.robot_plot.set_data([], [])
        return self.path_plot, self.robot_plot

    def update_anim(self, frame, total_force_func, compute_field_func, dt, obstacle_speeds):
        # 这里直接操作全局变量，后续可进一步封装
        global robot_position, obstacles, path_data, quiver_all, quiver_robot, im
        force = total_force_func(robot_position, goal, obstacles, d0, k_att, k_rep)
        robot_position[:] = robot_position + force * dt
        path_data.append(robot_position.copy())
        # 更新障碍物
        obstacles[:, 0] += obstacle_speeds[:, 0] * np.sin(0.05 * frame)
        obstacles[:, 1] += obstacle_speeds[:, 1] * np.cos(0.05 * frame)
        for obs_idx, obs in enumerate(obstacles):
            if obs[0] < x_limit[0] or obs[0] > x_limit[1]:
                obstacles[obs_idx, 0] = np.clip(obs[0], x_limit[0], x_limit[1])
                obstacle_speeds[obs_idx, 0] *= -1
            if obs[1] < y_limit[0] or obs[1] > y_limit[1]:
                obstacles[obs_idx, 1] = np.clip(obs[1], y_limit[0], y_limit[1])
                obstacle_speeds[obs_idx, 1] *= -1
        # 重新计算势场和力场
        Z, U, V = compute_field_func(goal, obstacles, k_att, k_rep, d0)
        self.im.set_data(Z)
        self.quiver_all.set_UVC(U, V)
        path = np.array(path_data)
        self.path_plot.set_data(path[:, 0], path[:, 1])
        self.robot_plot.set_data([robot_position[0]], [robot_position[1]])
        self.obstacles_plot.set_data(obstacles[:, 0], obstacles[:, 1])
        self.quiver_robot.remove()
        self.quiver_robot = self.ax.quiver(robot_position[0], robot_position[1], force[0], force[1], color='red', scale=1, width=0.005)
        if np.linalg.norm(robot_position - goal) < 0.3:
            print(f"Goal reached at frame {frame}!")
        return self.path_plot, self.robot_plot, self.obstacles_plot, self.im, self.quiver_all

    def run(self, total_force_func, compute_field_func, dt, obstacle_speeds):
        if not self.show:
            return
        self.setup()
        self.ani = animation.FuncAnimation(
            self.fig, lambda frame: self.update_anim(frame, total_force_func, compute_field_func, dt, obstacle_speeds),
            frames=2000, init_func=self.init_anim, blit=False, interval=50, repeat=False)
        plt.show()

# ================= 参数集中管理与自动缩放 =================
class APFConfig:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.k_att = 0.01 / max(width, height)
        self.k_rep = 0.1 / max(width, height)
        self.d0 = 0.2 * min(width, height)  # 障碍物影响范围
        self.max_rep_force = 0.05 * max(width, height)
        self.dt = 0.01 * max(width, height)
        self.x_limit = (0, width)
        self.y_limit = (0, height)
        self.x_range = np.linspace(self.x_limit[0], self.x_limit[1], min(120, width//16))
        self.y_range = np.linspace(self.y_limit[0], self.y_limit[1], min(68, height//16))
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)

# ================= 势场/力场与环境解耦 =================
class APFField:
    def __init__(self, config: APFConfig):
        self.cfg = config

    def attractive_force(self, position, goal):
        return -self.cfg.k_att * (position - goal)

    def repulsive_force(self, position, obstacles):
        force = np.zeros(2)
        for obs in obstacles:
            diff = position - obs
            dist = np.linalg.norm(diff)
            if dist < self.cfg.d0:
                repulsive_force = self.cfg.k_rep * ((1/dist) - (1/self.cfg.d0)) * (1/(dist**2)) * (diff/dist)
                norm = np.linalg.norm(repulsive_force)
                if norm > self.cfg.max_rep_force:
                    repulsive_force = repulsive_force / norm * self.cfg.max_rep_force
                force += repulsive_force
        return force

    def total_force(self, position, goal, obstacles):
        force_att = self.attractive_force(position, goal)
        force_rep = self.repulsive_force(position, obstacles)
        return force_att + force_rep

    def compute_potential_and_force_field(self, goal, obstacles):
        X, Y = self.cfg.X, self.cfg.Y
        k_att, k_rep, d0, max_rep_force = self.cfg.k_att, self.cfg.k_rep, self.cfg.d0, self.cfg.max_rep_force
        diff_goal = np.stack([X - goal[0], Y - goal[1]], axis=-1)
        att_potential = 0.5 * k_att * np.sum(diff_goal**2, axis=-1)
        force_att = -k_att * diff_goal
        rep_potential = np.zeros_like(X)
        force_rep = np.zeros(X.shape + (2,))
        for obs in obstacles:
            diff_obs = np.stack([X - obs[0], Y - obs[1]], axis=-1)
            dist = np.linalg.norm(diff_obs, axis=-1)
            mask = dist < d0
            safe_dist = np.where(mask, dist, 1)
            rep_potential[mask] += 0.5 * k_rep * ((1/safe_dist[mask]) - (1/d0))**2
            coeff = k_rep * ((1/safe_dist[mask]) - (1/d0)) * (1/(safe_dist[mask]**2))
            coeff = coeff[:, np.newaxis]
            direction = diff_obs[mask] / safe_dist[mask][:, np.newaxis]
            rep_force = coeff * direction
            norm_rep = np.linalg.norm(rep_force, axis=-1)
            over = norm_rep > max_rep_force
            if np.any(over):
                rep_force[over] = rep_force[over] / norm_rep[over][:, np.newaxis] * max_rep_force
            force_rep[mask] += rep_force
        Z = att_potential + rep_potential
        U = force_att[..., 0] + force_rep[..., 0]
        V = force_att[..., 1] + force_rep[..., 1]
        return Z, U, V

# ================= 主流程示例 =================
if __name__ == '__main__':
    # 设定环境尺寸
    width, height = 1920, 1080
    cfg = APFConfig(width, height)
    apf = APFField(cfg)
    # 初始点、目标点、障碍物
    robot_position = np.array([width//2, height//2], dtype=float)
    goal = robot_position.copy()
    path_data = [robot_position.copy()]
    # 障碍物示例
    obstacles = np.array([[300, 200], [1500, 900], [900, 500]])
    obstacle_speeds = np.array([[0.1, 0.2], [-0.2, -0.1], [0.1, -0.1]])
    # 初始势场
    Z, U, V = apf.compute_potential_and_force_field(goal, obstacles)
    # 可视化
    show_visualization = True
    vis = Visualize(Z, U, V, cfg.X, cfg.Y, robot_position, goal, obstacles, path_data, cfg.x_limit, cfg.y_limit, show=show_visualization)
    vis.run(lambda pos, goal, obs, *_: apf.total_force(pos, goal, obs),
            lambda goal, obs, *_: apf.compute_potential_and_force_field(goal, obs),
            cfg.dt, obstacle_speeds)