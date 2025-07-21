# 合并函数，将MaskDictionaryModel中的所有mask合并为一个bool类型的ndarray

import numpy as np
import torch

from external.grounding_sam2.utils.mask_dictionary_model import MaskDictionaryModel

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
    
    # 收集所有有效的mask到GPU上进行批量处理
    valid_masks = []
    
    for obj_id, obj_info in mask_dict.labels.items():
        # 获取mask数据
        if torch.is_tensor(obj_info.mask):
            mask = obj_info.mask
        else:
            # 如果是numpy数组，转换为tensor并移到GPU
            mask = torch.from_numpy(obj_info.mask).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查形状匹配
        if mask.shape != target_shape:
            print(f"[Warning] Mask shape mismatch: expected {target_shape}, got {mask.shape}")
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
        merged_mask = merged_mask_gpu.cpu().numpy()
        
    except (RuntimeError, MemoryError) as e:
        # 如果GPU内存不足或出错，回退到CPU处理
        print(f"[Warning] GPU batch processing failed, falling back to CPU: {e}")
        merged_mask = np.zeros(target_shape, dtype=bool)
        
        for mask in valid_masks:
            # 逐个处理，确保数据在CPU上
            if torch.is_tensor(mask):
                mask_cpu = mask.cpu().numpy()
            else:
                mask_cpu = mask
            merged_mask = merged_mask | mask_cpu
    
    return merged_mask