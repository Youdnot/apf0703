# 人工势场法仿真系统 - 重构版本

## 概述

本项目实现了一个基于人工势场法（Artificial Potential Field, APF）的机器人路径规划仿真系统。经过重构后，代码结构更加清晰，配置管理更加集中。

## 代码结构

### 主要文件

- `main_force2ui_control.py` - 主程序文件
- `config.py` - 配置管理模块
- `utils/calculate_force.py` - 力计算工具模块
- `utils/convert_coordinate.py` - 坐标转换工具
- `utils/control_unity_ui.py` - Unity UI控制模块

### 配置系统

使用`@dataclass`装饰器封装了以下配置类：

#### ViewConfig
```python
@dataclass
class ViewConfig:
    width: int = 1920    # 视图宽度
    height: int = 1080   # 视图高度
```

#### PhysicsConfig
```python
@dataclass
class PhysicsConfig:
    k_att: float = 10.0          # 吸引力系数
    k_rep: float = 10.0          # 排斥力系数
    damping_factor: float = 1.0   # 阻尼系数
    d0: float = 200.0            # 障碍物影响范围
    max_v: float = 20.0          # 最大速度
    dt: float = 0.2              # 时间步长
```

#### SimulationConfig
```python
@dataclass
class SimulationConfig:
    anchor_point: np.ndarray      # 锚点位置
    init_pos: np.ndarray         # 初始位置
    init_vel: np.ndarray = None  # 初始速度
```

#### ObstacleConfig
```python
@dataclass
class ObstacleConfig:
    x_start: int = 500   # 障碍物起始x坐标
    x_end: int = 600     # 障碍物结束x坐标
    y_start: int = 600   # 障碍物起始y坐标
    y_end: int = 700     # 障碍物结束y坐标
```

### ConfigManager

`ConfigManager`类提供了统一的配置管理接口：

```python
class ConfigManager:
    def __init__(self):
        # 初始化所有配置
        self.view_config = ViewConfig()
        self.physics_config = PhysicsConfig()
        # ...
    
    def create_obstacle_mask(self) -> np.ndarray:
        """创建障碍物掩码"""
        
    def get_initial_position(self) -> np.ndarray:
        """获取调整后的初始位置"""
```

## 主要功能

### 1. 力计算
- `get_attractive_force()` - 计算吸引力
- `get_repulsive_force()` - 计算排斥力
- `get_total_force()` - 计算总力

### 2. 位置更新
- `update_position_and_velocity()` - 更新位置和速度
- `update_position_and_velocity_with_config()` - 使用配置对象更新位置和速度

### 3. 配置管理
- 集中管理所有仿真参数
- 提供便捷的配置访问接口
- 支持参数验证和默认值设置

## 使用方法

### 基本使用
```python
from config import config_manager
from utils.calculate_force import update_position_and_velocity_with_config

# 获取配置
view_config = config_manager.view_config
physics_config = config_manager.physics_config

# 创建障碍物
obstacle_mask = config_manager.create_obstacle_mask()

# 更新位置和速度
force, new_pos, new_vel, converted_pos, path_data = update_position_and_velocity_with_config(
    cur_pos, cur_vel, anchor_point, obstacle_mask, 
    view_config, physics_config, path_data
)
```

### 自定义配置
```python
# 修改物理参数
config_manager.physics_config.k_att = 15.0
config_manager.physics_config.k_rep = 12.0

# 修改视图参数
config_manager.view_config.width = 1600
config_manager.view_config.height = 900
```

## 优势

1. **代码清晰度** - 使用`@dataclass`使配置结构更加清晰
2. **参数集中管理** - 所有配置参数集中在`config.py`中
3. **易于维护** - 配置修改不需要在多个文件中查找
4. **类型安全** - 使用类型注解提高代码可靠性
5. **扩展性好** - 新增配置项只需在相应的dataclass中添加

## 运行

```bash
python main_force2ui_control.py
```

程序将启动人工势场法仿真，并在Unity UI中显示机器人的运动轨迹。 