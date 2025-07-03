import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import os

# -------------------- 配置参数 --------------------
ENV_W, ENV_H = 1920, 1080  # 环境分辨率
WIN_W, WIN_H = 800, 400    # 窗口尺寸
ANCHOR = (ENV_W // 2, ENV_H // 2)  # 锚定点
K_ATT = 1e-3   # 吸引力系数
K_REP = 5.0    # 排斥力系数
REP_RANGE = 200  # 排斥影响范围（像素）
DT = 1.0       # 时间步长
DAMPING = 0.85 # 阻尼系数

# -------------------- 随机障碍mask生成 --------------------
def random_mask(env_h, env_w, n_shapes=3):
    """生成不规则障碍mask（多椭圆/多边形叠加）"""
    mask = np.zeros((env_h, env_w), dtype=bool)
    for _ in range(n_shapes):
        # 随机椭圆参数
        cy = np.random.randint(100, env_h-100)
        cx = np.random.randint(100, env_w-100)
        ry = np.random.randint(40, 120)
        rx = np.random.randint(40, 120)
        y, x = np.ogrid[:env_h, :env_w]
        ellipse = ((y-cy)/ry)**2 + ((x-cx)/rx)**2 <= 1
        mask |= ellipse
    return mask

def load_mask_from_npy(path):
    """从npy文件读取mask，要求shape=(1080,1920)且为bool类型"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")
    mask = np.load(path)
    assert mask.shape == (ENV_H, ENV_W) and mask.dtype == bool, "mask格式错误"
    return mask

# -------------------- 势场生成 --------------------
def compute_potential_field(mask, anchor, k_att, k_rep, rep_range):
    """生成势场：吸引+排斥"""
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    # 吸引势场
    att = 0.5 * k_att * ((x - anchor[0])**2 + (y - anchor[1])**2)
    # 排斥势场
    rep = np.zeros_like(att)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~mask)
    rep_mask = (dist < rep_range)
    with np.errstate(divide='ignore'):
        rep[rep_mask] = 0.5 * k_rep * ((1/dist[rep_mask] - 1/rep_range)**2)
    return att + rep, dist

# -------------------- 窗口相关 --------------------
def window_mask(center, win_w, win_h, env_h, env_w):
    """返回窗口覆盖的像素mask"""
    cx, cy = int(center[0]), int(center[1])
    x0 = max(0, cx - win_w//2)
    x1 = min(env_w, cx + win_w//2)
    y0 = max(0, cy - win_h//2)
    y1 = min(env_h, cy + win_h//2)
    mask = np.zeros((env_h, env_w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask

# -------------------- 力的计算 --------------------
def compute_force(potential, center, win_w, win_h):
    """在窗口区域内采样梯度，取均值作为窗口受力"""
    y0 = max(0, int(center[1] - win_h // 2))
    y1 = min(potential.shape[0], int(center[1] + win_h // 2))
    x0 = max(0, int(center[0] - win_w // 2))
    x1 = min(potential.shape[1], int(center[0] + win_w // 2))
    # 区域至少1x1
    if y1 <= y0: y1 = y0 + 1
    if x1 <= x0: x1 = x0 + 1
    grad_y, grad_x = np.gradient(potential)
    fx = -np.mean(grad_x[y0:y1, x0:x1])
    fy = -np.mean(grad_y[y0:y1, x0:x1])
    # 防止NaN
    if np.isnan(fx) or np.isnan(fy):
        fx, fy = 0.0, 0.0
    return np.array([fx, fy])

# -------------------- 最低点查找 --------------------
def find_minimum_point(potential):
    """返回势场最低点坐标"""
    idx = np.unravel_index(np.argmin(potential), potential.shape)
    return (idx[1], idx[0])  # (x, y)

# -------------------- 主流程与可视化 --------------------
def main(mask_source=None):
    # 初始化mask
    if mask_source is None:
        mask = random_mask(ENV_H, ENV_W)
        print("[INFO] 使用随机障碍mask")
    elif mask_source.endswith('.npy'):
        mask = load_mask_from_npy(mask_source)
        print(f"[INFO] 从文件加载mask: {mask_source}")
    else:
        raise ValueError("mask_source必须为None或npy文件路径")
    print(f"[DEBUG] mask sum: {mask.sum()}")

    # 初始化窗口
    window_center = np.array(ANCHOR, dtype=float)
    window_velocity = np.zeros(2)

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
    ax.set_xlim(0, ENV_W)
    ax.set_ylim(ENV_H, 0)
    ax.set_aspect('equal')
    ax.set_title('AR Window Adaptive Movement with Potential Field')

    # 绘制初始势场
    potential, dist = compute_potential_field(mask, ANCHOR, K_ATT, K_REP, REP_RANGE)
    im = ax.imshow(potential, cmap='viridis', alpha=0.7, extent=[0, ENV_W, ENV_H, 0])
    # 绘制障碍mask
    mask_img = ax.imshow(mask, cmap='Reds', alpha=0.3, extent=[0, ENV_W, ENV_H, 0])
    # 绘制窗口
    win_rect = Rectangle((window_center[0]-WIN_W/2, window_center[1]-WIN_H/2), WIN_W, WIN_H,
                        linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(win_rect)
    # 绘制窗口中心
    center_dot, = ax.plot([window_center[0]], [window_center[1]], 'bo', markersize=8)
    # 绘制势场最低点
    min_pt = find_minimum_point(potential)
    min_dot, = ax.plot([min_pt[0]], [min_pt[1]], 'go', markersize=10)
    # 绘制窗口移动方向箭头
    arrow = ax.arrow(window_center[0], window_center[1], 0, 0, color='red', width=5, head_width=30)

    def update(frame):
        nonlocal mask, window_center, window_velocity, potential, dist, min_pt, arrow
        # 随机动态障碍mask（可替换为外部输入）
        if frame % 100 == 0:
            mask = random_mask(ENV_H, ENV_W)
            print(f"[INFO] frame {frame}: mask sum {mask.sum()}")
        # 势场与距离
        potential, dist = compute_potential_field(mask, ANCHOR, K_ATT, K_REP, REP_RANGE)
        # 找最低点
        min_pt = find_minimum_point(potential)
        # 计算窗口中心点处的力
        force = compute_force(potential, window_center, WIN_W, WIN_H)
        # 过阻尼运动
        window_velocity = (window_velocity + force * DT) * DAMPING
        window_center += window_velocity * DT
        # 限制窗口中心点在画面内，防止NaN
        window_center[0] = np.clip(window_center[0], WIN_W//2, ENV_W - WIN_W//2)
        window_center[1] = np.clip(window_center[1], WIN_H//2, ENV_H - WIN_H//2)
        # 检查NaN
        if np.isnan(window_center[0]) or np.isnan(window_center[1]):
            print('[ERROR] window_center NaN, 重置为锚点')
            window_center[:] = ANCHOR
            window_velocity[:] = 0
        # 可视化更新
        im.set_data(potential)
        mask_img.set_data(mask)
        win_rect.set_xy((window_center[0]-WIN_W/2, window_center[1]-WIN_H/2))
        center_dot.set_data([window_center[0]], [window_center[1]])
        min_dot.set_data([min_pt[0]], [min_pt[1]])
        # 更新箭头
        arrow.remove()
        arrow = ax.arrow(window_center[0], window_center[1], force[0]*30, force[1]*30, color='red', width=5, head_width=30)
        # 打印调试信息
        if frame % 50 == 0:
            print(f"[DEBUG] frame {frame}: win_center=({window_center[0]:.1f},{window_center[1]:.1f}), force=({force[0]:.2f},{force[1]:.2f}), v=({window_velocity[0]:.2f},{window_velocity[1]:.2f})")
        return im, mask_img, win_rect, center_dot, min_dot, arrow

    ani = animation.FuncAnimation(fig, update, frames=1000, interval=40, blit=False)
    plt.show()

if __name__ == '__main__':
    main() 