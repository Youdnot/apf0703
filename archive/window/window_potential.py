import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import random
from scipy.ndimage import binary_dilation

class ARWindow:
    """AR窗口类，管理窗口的位置、尺寸和运动状态"""
    
    def __init__(self, initial_position, size=(800, 400)):
        """
        初始化AR窗口
        
        Args:
            initial_position: tuple, 窗口初始中心位置 (x, y)
            size: tuple, 窗口尺寸 (width, height)
        """
        self.position = np.array(initial_position, dtype=float)
        self.size = np.array(size)
        self.anchor_point = np.array(initial_position, dtype=float)  # 锚定点
        self.velocity = np.zeros(2)  # 当前速度
        self.half_size = self.size / 2  # 半尺寸，用于碰撞检测
        
        print(f"AR Window initialized at position: {self.position}")
        print(f"AR Window size: {self.size}")
        print(f"AR Window anchor point: {self.anchor_point}")

class PotentialField:
    """势场类，管理环境势场的生成和更新"""
    
    def __init__(self, grid_size=(1920, 1080), anchor_point=None):
        """
        初始化势场
        
        Args:
            grid_size: tuple, 环境网格尺寸 (width, height)
            anchor_point: tuple, 锚定点位置
        """
        self.grid_size = grid_size
        self.width, self.height = grid_size
        self.field = np.zeros((self.height, self.width))  # 势场矩阵
        self.anchor_point = np.array(anchor_point) if anchor_point else np.array([960, 540])
        
        # 势场参数（可调节）
        self.k_att = 2.0  # 吸引力系数
        self.k_rep = 50.0  # 排斥力系数
        self.rep_range = 200.0  # 排斥力影响范围
        
        # 预计算坐标网格
        x_coords = np.arange(self.width)
        y_coords = np.arange(self.height)
        self.X, self.Y = np.meshgrid(x_coords, y_coords)
        
        print(f"Potential field initialized with grid size: {self.grid_size}")
        print(f"Anchor point at: {self.anchor_point}")
        
    def update_field(self, obstacle_mask=None):
        """
        更新势场分布（优化版本）
        
        Args:
            obstacle_mask: numpy array, 障碍物掩膜 (bool类型)
        """
        self.field.fill(0)  # 清空势场
        
        # 1. 添加锚定点的吸引势场
        dist_to_anchor = np.sqrt((self.X - self.anchor_point[0])**2 + 
                                (self.Y - self.anchor_point[1])**2)
        attractive_field = 0.5 * self.k_att * dist_to_anchor**2
        self.field += attractive_field
        
        # 2. 添加障碍物的排斥势场（优化版本）
        if obstacle_mask is not None:
            # 计算到所有障碍物的距离场
            from scipy.ndimage import distance_transform_edt
            
            # 计算到最近障碍物的距离
            distance_to_obstacles = distance_transform_edt(~obstacle_mask)
            
            # 只在排斥范围内计算排斥势场
            mask_in_range = distance_to_obstacles < self.rep_range
            
            # 避免除零错误
            dist_safe = np.where(distance_to_obstacles < 1.0, 1.0, distance_to_obstacles)
            
            repulsive_field = np.zeros_like(self.field)
            repulsive_field[mask_in_range] = (
                0.5 * self.k_rep * 
                ((1.0 / dist_safe[mask_in_range]) - (1.0 / self.rep_range))**2
            )
            
            self.field += repulsive_field
                
        print(f"Potential field updated. Min: {self.field.min():.2f}, Max: {self.field.max():.2f}")
        
    def find_minimum_potential(self, search_radius=None, center=None):
        """
        在指定区域内找到势场最低点
        
        Args:
            search_radius: float, 搜索半径
            center: tuple, 搜索中心点
            
        Returns:
            tuple: 最低势能点的坐标 (x, y)
        """
        if search_radius is None or center is None:
            # 全局搜索
            min_idx = np.unravel_index(np.argmin(self.field), self.field.shape)
            return (int(min_idx[1]), int(min_idx[0]))  # 转换为(x, y)坐标，确保为整数
        
        # 局部搜索
        cx, cy = int(center[0]), int(center[1])
        r = int(search_radius)
        
        y_min = max(0, cy - r)
        y_max = min(self.height, cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(self.width, cx + r + 1)
        
        # 确保搜索区域有效
        if y_max <= y_min or x_max <= x_min:
            return (int(center[0]), int(center[1]))
        
        local_field = self.field[y_min:y_max, x_min:x_max]
        local_min_idx = np.unravel_index(np.argmin(local_field), local_field.shape)
        
        return (int(x_min + local_min_idx[1]), int(y_min + local_min_idx[0]))

class AdaptiveWindowSystem:
    """AR自适应窗口系统主类"""
    
    def __init__(self, initial_position=(960, 540)):
        """
        初始化自适应窗口系统
        
        Args:
            initial_position: tuple, 窗口初始位置
        """
        self.window = ARWindow(initial_position)
        self.potential_field = PotentialField(anchor_point=initial_position)
        
        # 运动控制参数（可调节）
        self.damping_coefficient = 0.8  # 阻尼系数
        self.max_velocity = 50.0  # 最大速度限制
        self.force_threshold = 10.0  # 力的阈值，低于此值保持静止
        
        # 边界约束参数
        self.boundary_margin = 200  # 边界缓冲区
        self.boundary_force = 100.0  # 边界约束力强度
        
        # 模拟障碍物参数
        self.obstacles = []  # 存储当前障碍物
        self.obstacle_lifetime = []  # 障碍物生存时间
        self.frame_count = 0  # 帧计数器
        
        print(f"Adaptive window system initialized")
        print(f"Motion parameters - Damping: {self.damping_coefficient}, Max velocity: {self.max_velocity}")
        
    def generate_random_obstacles(self):
        """
        生成随机障碍物模拟动态mask输入
        """
        self.frame_count += 1
        
        # 随机生成新障碍物
        if random.random() < 0.02:  # 2%概率生成新障碍物
            obstacle = {
                'x': random.randint(200, 1720),
                'y': random.randint(200, 880),
                'width': random.randint(100, 300),
                'height': random.randint(100, 200)
            }
            self.obstacles.append(obstacle)
            self.obstacle_lifetime.append(random.randint(100, 300))  # 生存100-300帧
            print(f"New obstacle generated at ({obstacle['x']}, {obstacle['y']})")
        
        # 移除过期障碍物
        for i in range(len(self.obstacle_lifetime) - 1, -1, -1):
            self.obstacle_lifetime[i] -= 1
            if self.obstacle_lifetime[i] <= 0:
                removed_obs = self.obstacles.pop(i)
                self.obstacle_lifetime.pop(i)
                print(f"Obstacle removed from ({removed_obs['x']}, {removed_obs['y']})")
        
        print(f"Current obstacles count: {len(self.obstacles)}")
        
    def create_obstacle_mask(self):
        """
        根据当前障碍物创建掩膜
        
        Returns:
            numpy array: bool类型的障碍物掩膜
        """
        mask = np.zeros((self.potential_field.height, self.potential_field.width), dtype=bool)
        
        for obs in self.obstacles:
            x1 = max(0, obs['x'])
            y1 = max(0, obs['y'])
            x2 = min(self.potential_field.width, obs['x'] + obs['width'])
            y2 = min(self.potential_field.height, obs['y'] + obs['height'])
            
            if x2 > x1 and y2 > y1:  # 确保有效区域
                mask[y1:y2, x1:x2] = True
        
        # 考虑窗口尺寸，对mask进行适度膨胀处理（配置空间）
        if len(self.obstacles) > 0:
            dilation_size = max(1, min(5, int(max(self.window.half_size) // 50)))  # 限制膨胀尺寸
            structure = np.ones((dilation_size, dilation_size))
            try:
                mask = binary_dilation(mask, structure=structure)
            except Exception as e:
                print(f"Dilation error: {e}")
                # 如果膨胀失败，使用原始mask
                pass
        
        return mask
        
    def calculate_window_obstacle_overlap(self, obstacle_mask):
        """
        计算窗口与障碍物的重叠情况
        
        Args:
            obstacle_mask: numpy array, 障碍物掩膜
            
        Returns:
            float: 重叠面积比例 (IOU)
        """
        wx, wy = int(self.window.position[0]), int(self.window.position[1])
        hw, hh = int(self.window.half_size[0]), int(self.window.half_size[1])
        
        x1 = max(0, wx - hw)
        y1 = max(0, wy - hh)
        x2 = min(self.potential_field.width, wx + hw)
        y2 = min(self.potential_field.height, wy + hh)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        window_area = (x2 - x1) * (y2 - y1)
        overlap_area = int(np.sum(obstacle_mask[y1:y2, x1:x2]))
        
        iou = overlap_area / window_area if window_area > 0 else 0.0
        return float(iou)
        
    def calculate_forces(self, obstacle_mask):
        """
        计算作用在窗口上的总力
        
        Args:
            obstacle_mask: numpy array, 障碍物掩膜
            
        Returns:
            numpy array: 总力向量 (fx, fy)
        """
        # 更新势场
        self.potential_field.update_field(obstacle_mask)
        
        # 计算重叠情况
        overlap_ratio = self.calculate_window_obstacle_overlap(obstacle_mask)
        
        # 确定目标点
        if overlap_ratio > 0.1 or len(self.obstacles) > 0:
            # 有重叠或有障碍物时，目标为势场最低点
            target = self.potential_field.find_minimum_potential(
                search_radius=300, center=self.window.position)
        else:
            # 无障碍物时，目标为锚定点
            target = self.window.anchor_point
        
        target = np.array(target)
        
        # 计算吸引力
        direction = target - self.window.position
        distance = np.linalg.norm(direction)
        
        if distance < 1.0:
            attractive_force = np.zeros(2)
        else:
            attractive_force = self.potential_field.k_att * direction
        
        # 添加边界约束力
        boundary_force = self.calculate_boundary_force()
        
        total_force = attractive_force + boundary_force
        
        # 应用力的阈值
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude < self.force_threshold:
            total_force = np.zeros(2)
        
        print(f"Overlap ratio: {overlap_ratio:.3f}, Target: ({target[0]:.1f}, {target[1]:.1f})")
        print(f"Force magnitude: {force_magnitude:.2f}")
        
        return total_force
        
    def calculate_boundary_force(self):
        """
        计算边界约束力
        
        Returns:
            numpy array: 边界约束力向量
        """
        force = np.zeros(2)
        pos = self.window.position
        
        # 考虑窗口半尺寸的边界约束
        margin = self.boundary_margin
        
        if pos[0] < margin:
            force[0] += self.boundary_force * (margin - pos[0]) / margin
        elif pos[0] > self.potential_field.width - margin:
            force[0] -= self.boundary_force * (pos[0] - (self.potential_field.width - margin)) / margin
            
        if pos[1] < margin:
            force[1] += self.boundary_force * (margin - pos[1]) / margin
        elif pos[1] > self.potential_field.height - margin:
            force[1] -= self.boundary_force * (pos[1] - (self.potential_field.height - margin)) / margin
            
        return force
        
    def update_window_position(self, dt=0.1):
        """
        使用过阻尼运动模型更新窗口位置
        
        Args:
            dt: float, 时间步长
        """
        # 生成障碍物
        self.generate_random_obstacles()
        
        # 创建障碍物掩膜
        obstacle_mask = self.create_obstacle_mask()
        
        # 计算力
        force = self.calculate_forces(obstacle_mask)
        
        # 过阻尼运动模型: ma = F - bv (其中b为阻尼系数)
        # 简化为: v_new = (v_old + F*dt) * damping
        self.window.velocity = (self.window.velocity + force * dt) * self.damping_coefficient
        
        # 限制最大速度
        velocity_magnitude = np.linalg.norm(self.window.velocity)
        if velocity_magnitude > self.max_velocity:
            self.window.velocity = (self.window.velocity / velocity_magnitude) * self.max_velocity
        
        # 更新位置
        self.window.position += self.window.velocity * dt
        
        print(f"Window position: ({self.window.position[0]:.1f}, {self.window.position[1]:.1f})")
        print(f"Window velocity: ({self.window.velocity[0]:.2f}, {self.window.velocity[1]:.2f})")

# 可视化和仿真代码
def create_visualization():
    """创建可视化界面"""
    
    # 初始化系统
    system = AdaptiveWindowSystem(initial_position=(960, 540))
    
    # 设置图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 主视图 - 显示窗口和障碍物
    ax1.set_xlim(0, 1920)
    ax1.set_ylim(0, 1080)
    ax1.set_aspect('equal')
    ax1.set_title('AR Window Adaptive Movement')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.invert_yaxis()  # 翻转Y轴以匹配图像坐标系
    
    # 势场视图
    ax2.set_xlim(0, 1920)
    ax2.set_ylim(0, 1080)
    ax2.set_aspect('equal')
    ax2.set_title('Potential Field Visualization')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.invert_yaxis()  # 翻转Y轴以匹配图像坐标系
    
    # 初始化绘图元素
    window_rect = Rectangle((0, 0), 800, 400, linewidth=2, edgecolor='blue', 
                           facecolor='lightblue', alpha=0.7)
    ax1.add_patch(window_rect)
    
    anchor_point, = ax1.plot([], [], 'go', markersize=10, label='Anchor Point')
    window_center, = ax1.plot([], [], 'bo', markersize=8, label='Window Center')
    path_line, = ax1.plot([], [], 'b-', alpha=0.5, label='Path')
    
    # 存储路径数据
    path_data = []
    obstacle_patches = []
    
    # 势场图像对象
    field_image = None
    
    def update_frame(frame):
        """更新动画帧"""
        nonlocal field_image
        
        # 更新系统状态
        system.update_window_position(dt=0.1)
        
        # 记录路径
        path_data.append(system.window.position.copy())
        if len(path_data) > 500:  # 限制路径长度
            path_data.pop(0)
        
        # 更新窗口位置
        wx, wy = system.window.position
        hw, hh = system.window.half_size
        window_rect.set_xy((wx - hw, wy - hh))
        
        # 更新其他元素
        anchor_point.set_data([system.window.anchor_point[0]], [system.window.anchor_point[1]])
        window_center.set_data([wx], [wy])
        
        if len(path_data) > 1:
            path_array = np.array(path_data)
            path_line.set_data(path_array[:, 0], path_array[:, 1])
        
        # 清除旧的障碍物
        for patch in obstacle_patches:
            patch.remove()
        obstacle_patches.clear()
        
        # 绘制新的障碍物
        for obs in system.obstacles:
            obs_rect = Rectangle((obs['x'], obs['y']), obs['width'], obs['height'],
                               facecolor='red', alpha=0.6)
            ax1.add_patch(obs_rect)
            obstacle_patches.append(obs_rect)
        
        # 更新势场可视化（始终显示）
        # 创建低分辨率的势场用于可视化
        downsample_factor = 10
        field_viz = system.potential_field.field[::downsample_factor, ::downsample_factor]
        extent = [0, 1920, 1080, 0]  # 注意Y轴范围顺序
        
        # 清除之前的势场图像
        if field_image is not None:
            field_image.remove()
        
        # 绘制新的势场
        field_image = ax2.imshow(field_viz, extent=extent, cmap='viridis', alpha=0.8, aspect='equal')
        
        # 在势场图上也显示窗口和障碍物
        # 清除之前的patches
        for patch in ax2.patches[:]:
            patch.remove()
            
        window_rect_field = Rectangle((wx - hw, wy - hh), 800, 400,
                                    linewidth=2, edgecolor='white', 
                                    facecolor='none')
        ax2.add_patch(window_rect_field)
        
        # 显示锚定点
        ax2.plot(system.window.anchor_point[0], system.window.anchor_point[1], 
                'go', markersize=8)
        
        for obs in system.obstacles:
            obs_rect_field = Rectangle((obs['x'], obs['y']), obs['width'], obs['height'],
                                     facecolor='red', alpha=0.8)
            ax2.add_patch(obs_rect_field)
        
        return [window_rect, anchor_point, window_center, path_line] + obstacle_patches
    
    # 设置图例
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update_frame, frames=2000, 
                                 interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting AR Adaptive Window System...")
    create_visualization()