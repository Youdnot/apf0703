"""
配置管理模块
集中管理系统的所有配置参数
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class ViewConfig:
    """视图配置参数"""
    width: int = 1920
    height: int = 1080

@dataclass
class WindowConfig:
    """窗口配置参数"""
    width: int = 200
    height: int = 200

@dataclass
class PhysicsConfig:
    """物理参数配置"""
    k_att: float = 10.0  # 吸引力系数
    k_rep: float = 10.0  # 排斥力系数
    damping_factor: float = 1.0  # 阻尼系数
    d0: float = 200.0  # 障碍物影响范围
    max_v: float = 10.0  # 最大速度
    dt: float = 0.1  # 时间步长

@dataclass
class SimulationConfig:
    """仿真配置参数"""
    anchor_point: np.ndarray = np.array([500, 700])
    init_pos: np.ndarray = np.array([500, 700])
    init_vel: np.ndarray = np.array([0, 0])

@dataclass
class HololensConfig:
    """PV配置参数"""
    # host: str = '192.168.31.89'
    host: str = '169.254.10.1'

@dataclass
class UIConfig:
    """UI配置参数"""
    # Position in camera space (x, y, z)
    position = [0, 0, 0.5]

    # Rotation in camera space (x, y, z, w) as a quaternion
    rotation = [0, 0, 0, 1]

    # Scale (x, y, z) in meters
    scale = 0.05
    scale = [scale, scale, 1]
    # ratio = 1920/1080
    # scale = [ratio*scale, scale, 1]

    # Texture file (must be jpg or png)
    texture_file = 'assets/texture.jpg'
    # texture_file = 'grid.png'

from datetime import datetime

@dataclass
class DetectionConfig:
    """检测配置参数"""
    output_dir: str = "outputs/" + datetime.now().strftime("%Y-%m-%d-%H%M")
    detection_interval: int = 10
    init_prompt_text: str = "hand."
    final_prompt_text: str = "obstacle. person. viehicle. car. bus. truck. desk. table. chair. bin."


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.view_config = ViewConfig()
        self.window_config = WindowConfig()
        self.physics_config = PhysicsConfig()
        self.sim_config = SimulationConfig()
        self.hololens_config = HololensConfig()
        self.ui_config = UIConfig()
        self.detection_config = DetectionConfig()
# 创建全局配置实例
config_manager = ConfigManager() 