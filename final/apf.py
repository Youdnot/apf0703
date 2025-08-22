import numpy as np
from multiprocessing import Queue
import rerun as rr

class APFCalculator:

    def __init__(self, anchor, position, obstacle_mask):
        self.anchor = anchor
        self.position = position
        self.velocity = 0
        self.obstacle_mask = obstacle_mask

        # Mapping
        # 1080p
        # self.scaling_factor = 0.1/338
        # self.center_x = 940
        # self.center_y = 576
        # 720p
        self.scaling_factor = 0.1/226
        self.center_x = 618
        self.center_y = 329

        self.depth = 0.5

        # Init inner variables
        self.force = None
        self.converted_pos = None

        # Physics
        self.k_att: float = 0.4  # 吸引力系数
        self.k_rep: float = 30.0  # 排斥力系数
        self.damping_factor: float = 1  # 阻尼系数
        self.d0: float = 400.0  # 障碍物影响范围
        self.max_v: float = 15.0  # 最大速度
        self.dt: float = 0.1  # 时间步长

        self.att_limit = 40

        self.rep_limit = 100

        # 处理停滞 - 如果周围障碍物密度过高，添加小扰动
        self.rep_density_threshold = 0.9    # 密度阈值
        self.rep_small_magnitude = 0.1      # 可调整扰动幅度

        # Cache
        # for repulsive force
        x_coords, y_coords = np.mgrid[0:1280, 0:720]
        self.x_coords = x_coords
        self.y_coords = y_coords

    def _convert_coordinates(self, position):
        """
        Convert position from camera space to Unity camera space.
        从左下角为原点的投影平面坐标系转换为
        以中心点为原点的Unity Camera Space
        并完成scale的缩放 pixel to factor
        """
        view_position = position.copy()
        # Adjust to center of camera space
        view_position -= np.array([self.center_x, self.center_y])
        # Adjust amplitude from image pixel space to Unity camera space
        view_position = view_position.astype(np.float32)
        view_position *= self.scaling_factor
        # Add depth to the z-axis
        view_position = np.append(view_position, self.depth)
        return view_position
    
    def get_attractive_force(self):
        attractive_force = self.anchor - self.position
        modulus = np.linalg.norm(attractive_force)

        # limit the force
        if modulus > self.att_limit:
            attractive_force = attractive_force*(self.att_limit/modulus)
        
        return attractive_force
    
    def get_repulsive_force(self, obstacle_mask):
        if obstacle_mask.any():
            # 计算到当前位置和锚点的距离矩阵
            window_distances = np.sqrt((self.x_coords - self.position[0])**2 + (self.y_coords - self.position[1])**2)
            anchor_distances = np.sqrt((self.x_coords - self.anchor[0])**2 + (self.y_coords - self.anchor[1])**2)
            
            # 避免除零错误，设置最小距离
            window_distances = np.maximum(window_distances, 1)
            anchor_distances = np.maximum(anchor_distances, 1)
            
            # 计算影响范围掩码
            window_influence_mask = window_distances <= self.d0
            anchor_influence_mask = anchor_distances <= self.d0
            # 按位或进行合并
            influence_mask = window_influence_mask | anchor_influence_mask
            # 按位与进行筛选真正起作用的obstacles
            final_mask = obstacle_mask & influence_mask
            
            # 计算排斥力系数：repulsive_coefficient = (1/D - 1/d0) * (1/D)^2
            inv_distances = 1.0 / window_distances
            inv_d0 = 1.0 / self.d0
            repulsive_coefficient = (inv_distances - inv_d0) * (inv_distances ** 2)
            
            # 只在有效障碍物位置应用排斥力
            repulsive_coefficient = np.where(final_mask, repulsive_coefficient, 0)
            
            # 计算每个网格到窗口点的距离矩阵
            # 这里减完有1单位的误差？索引从0开始
            window_distances_x = self.position[0] - self.x_coords
            window_distances_y = self.position[1] - self.y_coords

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
            
            if obstacle_density > self.rep_density_threshold:
                # 计算吸引力的单位向量作为定向扰动
                attractive_force = self.get_attractive_force()
                # 添加定向扰动
                # 这里取负号是因为要反向扰动，因为吸引力是向锚点方向的
                perturbation = -attractive_force/np.linalg.norm(attractive_force)*self.rep_small_magnitude
                
                # 限制斥力幅度，避免过大
                repulsive_force *= 0.5
                
                repulsive_force += perturbation
            
            modulus = np.linalg.norm(repulsive_force)

            # limit the force
            if modulus > self.rep_limit:
                repulsive_force = repulsive_force*(self.rep_limit/modulus)

        else:
            repulsive_force = np.zeros(2)
        
        return repulsive_force
    
    def get_total_force(self, obstacle_mask):
        attractive_force = self.get_attractive_force()
        repulsive_force = self.get_repulsive_force(obstacle_mask)
        # print(f"Attractive Force: {attractive_force}, Repulsive Force: {repulsive_force}")
        total_force = self.k_att * attractive_force + self.k_rep * repulsive_force
        return total_force
    
    def update_position_and_velocity(self, obstacle_mask):  
        # 计算当前力
        self.force = self.get_total_force(obstacle_mask)
        # print(f"Force: {force}, Force Norm: {np.linalg.norm(force)}")
        
        self.velocity = self.damping_factor * self.force + (1 - self.damping_factor) * self.velocity
        self.position = self.position + (self.velocity * self.max_v * self.dt)

        # test convertion
        self.converted_pos = self._convert_coordinates(self.position)

        # print(f"Current Position: {cur_pos}, Current Velocity: {cur_vel}, Converted Position: {converted_pos}")

    # @rr.shutdown_at_exit
    # def log_data(self):

    #     rr.init("Unified")
    #     rr.connect_grpc()

    def run(self):

        self.update_position_and_velocity(self.obstacle_mask)

        return self.converted_pos