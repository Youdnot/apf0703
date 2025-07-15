"""
配置管理模块
集中管理仿真系统的所有配置参数
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

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.view_config = ViewConfig()
        self.window_config = WindowConfig()
        self.physics_config = PhysicsConfig()
        self.sim_config = SimulationConfig()
# 创建全局配置实例
config_manager = ConfigManager() 