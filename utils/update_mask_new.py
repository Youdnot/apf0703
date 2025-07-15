#------------------------------------------------------------------------------
# This script receives masks from the detection model and process them.
# The mask is a np.array with shape (1920, 1080) and dtype=bool.
#------------------------------------------------------------------------------

import numpy as np

# 全局变量用于历史缓存
_mask_history = []
_max_history_size = 5  # 可自定义的缓存数量
_accumulated_mask = None  # 缓存累积结果，避免重复计算

def set_max_history_size(size: int) -> None:
    """
    设置历史缓存的最大大小
    
    Args:
        size: 缓存大小，建议范围 3-10
            - 较小的值（3-5）：响应更快，阻尼效应较弱
            - 较大的值（6-10）：阻尼效应更强，但响应较慢
    
    Raises:
        ValueError: 当 size < 1 时抛出异常
    """
    global _max_history_size
    if size < 1:
        raise ValueError("缓存大小必须大于0")
    _max_history_size = size

def clear_mask_history() -> None:
    """
    清空历史缓存
    
    在以下情况下建议调用此函数：
    - 系统重启时
    - 检测模型重新初始化时
    - 需要重置阻尼效应时
    """
    global _mask_history, _accumulated_mask
    _mask_history.clear()
    _accumulated_mask = None

def get_mask_history_size() -> int:
    """
    获取当前历史缓存的大小
    
    Returns:
        int: 当前缓存中的 mask 数量
    """
    return len(_mask_history)

def update_mask(last_mask: np.ndarray, cur_mask: np.ndarray) -> np.ndarray:
    """
    基于时间窗口的 mask 更新，使用增量更新实现阻尼效应
    
    工作原理：
    1. 维护一个累积的mask结果，避免重复计算
    2. 使用增量更新：新mask直接与累积结果进行或运算
    3. 当移除最旧的mask时，重新计算累积结果
    4. 避免内存复制和重复的堆叠操作
    
    Args:
        last_mask: 上一个 mask (bool 类型)，用于兼容性，实际不使用
        cur_mask: 当前检测 mask (bool 类型)
    
    Returns:
        np.ndarray: 更新后的 mask (bool 类型)
    
    性能优化：
    - 避免重复的 np.stack 操作
    - 使用增量更新而非重新计算
    - 减少内存分配和复制
    """
    global _mask_history, _max_history_size, _accumulated_mask
    
    # 1. 将当前 mask 加入历史队列（避免复制）
    _mask_history.append(cur_mask)
    
    # 2. 如果队列超限，移除最旧的 mask
    if len(_mask_history) > _max_history_size:
        removed_mask = _mask_history.pop(0)
        # 需要重新计算累积结果
        _accumulated_mask = None
    
    # 3. 增量更新累积结果
    if _accumulated_mask is None:
        # 首次计算或需要重新计算
        if len(_mask_history) == 1:
            _accumulated_mask = _mask_history[0]
        else:
            # 使用就地操作避免额外的内存分配
            _accumulated_mask = _mask_history[0].copy()
            for mask in _mask_history[1:]:
                np.logical_or(_accumulated_mask, mask, out=_accumulated_mask)
    else:
        # 增量更新：直接将新mask与累积结果进行或运算
        np.logical_or(_accumulated_mask, cur_mask, out=_accumulated_mask)
    
    return _accumulated_mask


def test_mask_update():
    """
    测试 mask 更新功能
    """
    # 清空历史缓存
    clear_mask_history()
    set_max_history_size(3)
    
    # 创建测试 mask
    shape = (10, 10)  # 使用小尺寸便于测试
    
    # 测试1：第一个 mask
    mask1 = np.zeros(shape, dtype=bool)
    mask1[2:4, 2:4] = True  # 在中心位置设置障碍物
    
    # 创建初始的 last_mask
    last_mask = np.zeros(shape, dtype=bool)
    result1 = update_mask(last_mask, mask1)
    print(f"测试1 - 缓存大小: {get_mask_history_size()}")
    print(f"结果1 中 True 的数量: {np.sum(result1)}")
    
    # 测试2：第二个 mask（部分重叠）
    mask2 = np.zeros(shape, dtype=bool)
    mask2[3:5, 3:5] = True  # 与第一个 mask 部分重叠
    
    result2 = update_mask(result1, mask2)
    print(f"测试2 - 缓存大小: {get_mask_history_size()}")
    print(f"结果2 中 True 的数量: {np.sum(result2)}")
    
    # 测试3：第三个 mask（新位置）
    mask3 = np.zeros(shape, dtype=bool)
    mask3[7:9, 7:9] = True  # 新位置
    
    result3 = update_mask(result2, mask3)
    print(f"测试3 - 缓存大小: {get_mask_history_size()}")
    print(f"结果3 中 True 的数量: {np.sum(result3)}")
    
    # 测试4：第四个 mask（超出缓存限制）
    mask4 = np.zeros(shape, dtype=bool)
    mask4[1:3, 1:3] = True  # 新位置
    
    result4 = update_mask(result3, mask4)
    print(f"测试4 - 缓存大小: {get_mask_history_size()}")
    print(f"结果4 中 True 的数量: {np.sum(result4)}")
    
    print("测试完成！")


if __name__ == "__main__":
    test_mask_update()