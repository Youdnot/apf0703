import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial.distance import cdist

# Setting parameters
# basic view
view_width = 1920
view_height = 1080

# anchor point
anchor_point = np.array([500, 700])


# window
window_width = 200
window_height = 200

init_pos = anchor_point.copy()
init_vel = np.array([0, 0])

max_v = 10

cur_pos = init_pos.copy()
cur_vel = init_vel.copy()

# obstacles
init_mask = np.zeros((view_width, view_height), dtype=bool)

def updateMask(cur_mask: np.ndarray, det_mask: np.ndarray) -> np.ndarray:
    """
    Update the mask of the window.
    """
    # 根据det_mask对cur_mask进行更新
    # 数据格式类似以下，从检测模型中输出的掩码获得，其中True表示障碍物，
    # 这是系统当前的mask，从状态中读取，需要
    cur_mask = np.zeros((view_width, view_height), dtype=bool)
    # 这是这一帧模型检测输出的mask，需要根据其对现有的障碍物进行更新
    det_mask = np.zeros((view_width, view_height), dtype=bool)
    # 根据det_mask对cur_mask进行动态更新，并保持一定的阻尼
    #cur_mask = det_mask...
    return cur_mask

def getObstacles(self) -> np.ndarray:
    """
    Get the obstacles of the window.
    """
    # 从mask中获取障碍物
    # obstacles = cur_mask...
    # 注意要兼容后续的力的计算
    return obstacles

# Setting parameters for the potential field
k_att = zeta =  10.0  # 吸引力系数
k_rep = eta = 10.0  # 排斥力系数



def getRepulsiveForce(self) -> np.ndarray:
    """
    Get the repulsive force of APF.

    Returns:
        rep_force (np.ndarray): repulsive force of APF
    """
    obstacles = np.array(list(self.obstacles))
    cur_pos = np.array([[self.robot.px, self.robot.py]])

    # 相较于原先只考虑机器人位置附近的障碍物，现在考虑了窗口（即原机器人）和锚点附近的障碍物
    D_window = cdist(obstacles, cur_pos)
    D_anchor = cdist(obstacles, anchor_point)
    rep_force = (1 / D_window - 1 / self.d_0) * (1 / D_window) ** 2 * (cur_pos - obstacles)
    
    valid_mask_window = np.argwhere((1 / D_window - 1 / self.d_0) > 0)[:, 0]
    valid_mask_anchor = np.argwhere((1 / D_anchor - 1 / self.d_0) > 0)[:, 0]
    valid_mask = valid_mask_window + valid_mask_anchor
    rep_force = np.sum(rep_force[valid_mask, :], axis=0)

    if not np.all(rep_force == 0):
        rep_force = rep_force / np.linalg.norm(rep_force)
    
    return rep_force

def getAttractiveForce(self, cur_pos: np.ndarray, anchor_pos: np.ndarray) -> np.ndarray:
        """
        Get the attractive force of APF.

        Parameters:
            cur_pos (np.ndarray): current position of window
            anchor_pos (np.ndarray): anchor position of window

        Returns
            attr_force (np.ndarray): attractive force
        """
        # 改为始终考虑对锚点的吸引力，而不是对目标点的吸引力
        attr_force = anchor_pos - cur_pos
        if not np.all(attr_force == 0):
            attr_force = attr_force / np.linalg.norm(attr_force)
        
        return attr_force

def updateWindow(self):
    """
    Update the window.
    """
    # 更新的过程
    cur_pos += cur_vel
    
    attr_force = getAttractiveForce(cur_pos, anchor_point)
    rep_force = getRepulsiveForce(cur_pos)
    cur_force = zeta * attr_force + eta * rep_force


    # compute desired velocity
    cur_vel += cur_force
    cur_vel /= np.linalg.norm(cur_vel)
    cur_vel *= max_v

    # 暂时不考虑对角度的控制，如归一化、转向等等，因为窗口没有机器人的机械转向限制


def visualize(self):
    """
    Visualize the window.
    """
    # 可视化的过程
    # 根据检测到的障碍物计算势场的分布
    pass

