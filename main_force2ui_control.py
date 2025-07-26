from core.calculate_force import *
from utils.convert_coordinate import *
from config import config_manager

# 获取配置
view_config = config_manager.view_config
window_config = config_manager.window_config
physics_config = config_manager.physics_config
sim_config = config_manager.sim_config

# 初始化路径数据
path_data = [sim_config.init_pos.copy()]

# 当前位置和速度
cur_pos = sim_config.init_pos.copy()
cur_vel = sim_config.init_vel.copy()

converted_pos = np.array([0, 0, 0])

# Initialize obstacle mask
obstacle_mask = np.zeros((view_config.width, view_config.height), dtype=bool)
obstacle_mask[300:750, 250:800] = True


#------------------------------------------------------------------------------

# UI control
from core.ui_control import *
from utils.keyboard_utils import *

# Initialize connection and create element
element_key = initialize_connection()

# 启动键盘监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

import time

# 主循环：更新位置和速度
try:
    for i in range(2000):
        # 检查停止事件是否被设置
        if stop_event.is_set():
            print("收到停止信号，退出循环")
            break
            
        # 更新位置和速度
        force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, sim_config.anchor_point, obstacle_mask, view_config.width, view_config.height, physics_config.d0, physics_config.k_att, physics_config.k_rep, physics_config.damping_factor, physics_config.max_v, physics_config.dt, path_data)
        update_position(converted_pos)
        
        time.sleep(0.1)  # 模拟时间延迟

except KeyboardInterrupt:
    print("程序被用户中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    print("正在清理资源...")
    # 停止监听器
    listener.stop()
    listener.join()
    
    # 断开连接
    disconnect()
    print("程序已结束")