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
    
    merged_mask = torch.zeros((mask_dict.mask_height, mask_dict.mask_width), dtype=torch.bool)
    
    for obj_id, obj_info in mask_dict.labels.items():
        merged_mask = merged_mask | obj_info.mask
    
    return merged_mask.cpu().numpy()


# 使用示例
if __name__ == "__main__":
    # 示例：如何在tracking_unit_test.py中使用
    """
    # 在tracking_unit_test.py中添加以下代码：
    
    from simple_mask_merger import get_merged_bool_mask
    
    # 在main函数中，获取current_mask_dict后：
    current_mask_dict = tracker.last_mask_dict
    
    # 合并所有mask
    merged_bool_mask = get_merged_bool_mask(current_mask_dict)
    
    # 使用合并后的mask
    print(f"合并mask形状: {merged_bool_mask.shape}")
    print(f"检测到的像素数: {np.sum(merged_bool_mask)}")
    """ 