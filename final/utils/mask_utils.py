# 合并函数，将MaskDictionaryModel中的所有mask合并为一个bool类型的ndarray

import numpy as np
import torch

from grounding_sam2.utils.mask_dictionary_model import MaskDictionaryModel

def get_merged_bool_mask(mask_dict: MaskDictionaryModel) -> np.ndarray:
    """
    将MaskDictionaryModel中的所有mask合并为一个bool类型的ndarray
    
    Args:
        mask_dict: MaskDictionaryModel实例
        
    Returns:
        np.ndarray: 合并后的bool mask，形状为(mask_height, mask_width)
    """
    if not mask_dict.labels:
        return np.zeros((mask_dict.mask_height, mask_dict.mask_width), dtype=bool)
    
    # 获取目标形状
    target_shape = (mask_dict.mask_height, mask_dict.mask_width)
    
    # 统一设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 收集所有有效的mask到统一设备进行批量处理
    valid_masks = []
    
    for obj_id, obj_info in mask_dict.labels.items():
        # 获取mask数据，转换为统一设备、统一dtype
        if torch.is_tensor(obj_info.mask):
            mask = obj_info.mask.to(device)
        else:
            mask = torch.from_numpy(obj_info.mask).to(device)
        mask = mask.bool()
        
        # 检查形状匹配
        if mask.shape != target_shape:
            print(f"[Warning] Mask shape mismatch: expected {target_shape}, got {mask.shape}")
            continue

        # 筛选小面积mask（mask为bool，sum即为像素计数）
        if mask.sum().item() < 8000:
            continue
        
        valid_masks.append(mask)
    
    # 如果没有有效mask，返回零数组
    if not valid_masks:
        return np.zeros(target_shape, dtype=bool)
    
    # 批量合并：在GPU上一次性处理所有mask
    try:
        # 将所有mask堆叠成3D张量 (N, H, W)
        stacked_masks = torch.stack(valid_masks, dim=0)
        # 在GPU上进行布尔或运算，一次性合并所有mask
        merged_mask_gpu = torch.any(stacked_masks, dim=0)
        # 一次性传输到CPU并转换为numpy
        merged_mask = merged_mask_gpu.detach().cpu().numpy()
        
    except (RuntimeError, MemoryError) as e:
        # 如果GPU内存不足或出错，回退到CPU处理
        print(f"[Warning] GPU batch processing failed, falling back to CPU: {e}")
        merged_mask = np.zeros(target_shape, dtype=bool)
        
        for mask in valid_masks:
            # 逐个处理，确保数据在CPU上
            mask_cpu = mask.detach().cpu().numpy()
            merged_mask = merged_mask | mask_cpu
    
    return merged_mask

def get_merged_bool_mask_depth(mask_dict: MaskDictionaryModel, depth_map: np.ndarray) -> np.ndarray:
    """
    将MaskDictionaryModel中的所有mask合并为一个bool类型的ndarray
    增加深度筛选
    
    Args:
        mask_dict: MaskDictionaryModel实例
        
    Returns:
        np.ndarray: 合并后的bool mask，形状为(mask_height, mask_width)
    """
    if not mask_dict.labels:
        return np.zeros((mask_dict.mask_height, mask_dict.mask_width), dtype=bool)
    
    # 获取目标形状
    target_shape = (mask_dict.mask_height, mask_dict.mask_width)
    
    # 统一设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 收集所有有效的mask到统一设备进行批量处理
    valid_masks = []

    # 将深度图移动到相同设备进行计算，避免频繁CPU/GPU拷贝
    depth_tensor = torch.from_numpy(depth_map).to(device)
    
    for obj_id, obj_info in mask_dict.labels.items():
        # 获取mask数据，转换为统一设备、统一dtype
        if torch.is_tensor(obj_info.mask):
            mask = obj_info.mask.to(device)
        else:
            mask = torch.from_numpy(obj_info.mask).to(device)
        mask = mask.bool()
        
        # 检查形状匹配
        if mask.shape != target_shape:
            print(f"[Warning] Mask shape mismatch: expected {target_shape}, got {mask.shape}")
            continue

        # 筛选小面积mask（mask为bool，sum即为像素计数）
        if mask.sum().item() < 8000:
            continue
        
        # 深度筛选：在统一设备上进行，忽略无效深度（<=0 或 非有限值）
        masked_depths = depth_tensor[mask]
        valid_depths = masked_depths[torch.isfinite(masked_depths) & (masked_depths > 0)]
        if valid_depths.numel() == 0:
            print("[Warning] No valid depths under mask; skipping object")
            continue
        if valid_depths.min().item() > 1.0:
            print(f"[Warning] Depth mask is too far: {valid_depths.min().item():.3f} m")
            continue
        
        valid_masks.append(mask)
    
    # 如果没有有效mask，返回零数组
    if not valid_masks:
        return np.zeros(target_shape, dtype=bool)
    
    # 批量合并：在GPU上一次性处理所有mask
    try:
        # 将所有mask堆叠成3D张量 (N, H, W)
        stacked_masks = torch.stack(valid_masks, dim=0)
        # 在GPU上进行布尔或运算，一次性合并所有mask
        merged_mask_gpu = torch.any(stacked_masks, dim=0)
        # 一次性传输到CPU并转换为numpy
        merged_mask = merged_mask_gpu.detach().cpu().numpy()
        
    except (RuntimeError, MemoryError) as e:
        # 如果GPU内存不足或出错，回退到CPU处理
        print(f"[Warning] GPU batch processing failed, falling back to CPU: {e}")
        merged_mask = np.zeros(target_shape, dtype=bool)
        
        for mask in valid_masks:
            # 逐个处理，确保数据在CPU上
            mask_cpu = mask.detach().cpu().numpy()
            merged_mask = merged_mask | mask_cpu
    
    return merged_mask