import numpy as np
import torch
from typing import Dict, List, Optional, Union
from external.grounding_sam2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo


class ExtendedMaskDictionaryModel(MaskDictionaryModel):
    """
    扩展的MaskDictionaryModel类，提供灵活的mask数据返回方式
    """
    
    def get_masks_by_class(self, class_name: str) -> dict:
        """
        获取指定类别的所有对象掩码
        
        Args:
            class_name: 类别名称 (如 "hand", "person")
        
        Returns:
            dict: {
                "class_name": str,
                "objects": [
                    {
                        "instance_id": int,
                        "mask": torch.Tensor,  # (H, W) bool tensor
                        "bbox": [x1, y1, x2, y2],
                        "logit": float,
                        "area": int
                    },
                    ...
                ]
            }
        """
        if not self.labels:
            return {"class_name": class_name, "objects": []}
        
        objects = []
        for obj_id, obj_info in self.labels.items():
            if obj_info.class_name == class_name:
                # 计算面积
                area = obj_info.mask.sum().item()
                
                objects.append({
                    "instance_id": obj_id,
                    "mask": obj_info.mask,  # 原始tensor
                    "bbox": [obj_info.x1, obj_info.y1, obj_info.x2, obj_info.y2],
                    "logit": obj_info.logit,
                    "area": area
                })
        
        return {
            "class_name": class_name,
            "objects": objects
        }

    def get_all_classes(self) -> dict:
        """
        获取所有类别的对象掩码
        
        Returns:
            dict: {
                "hand": {"class_name": "hand", "objects": [...]},
                "person": {"class_name": "person", "objects": [...]},
                ...
            }
        """
        class_groups = {}
        
        for obj_id, obj_info in self.labels.items():
            class_name = obj_info.class_name
            if class_name not in class_groups:
                class_groups[class_name] = {"class_name": class_name, "objects": []}
            
            area = obj_info.mask.sum().item()
            class_groups[class_name]["objects"].append({
                "instance_id": obj_id,
                "mask": obj_info.mask,
                "bbox": [obj_info.x1, obj_info.y1, obj_info.x2, obj_info.y2],
                "logit": obj_info.logit,
                "area": area
            })
        
        return class_groups

    def get_merged_mask(self, return_type: str = "numpy") -> dict:
        """
        获取合并后的整体mask
        
        Args:
            return_type: "numpy" 或 "tensor"
        
        Returns:
            dict: {
                "merged_mask": np.ndarray/torch.Tensor,  # (H, W) int32
                "metadata": {
                    "total_objects": int,
                    "classes_present": list,
                    "object_mapping": {instance_id: class_name},
                    "bbox_mapping": {instance_id: [x1, y1, x2, y2]}
                }
            }
        """
        if not self.labels:
            # 返回空mask
            empty_mask = np.zeros((self.mask_height, self.mask_width), dtype=np.int32)
            return {
                "merged_mask": empty_mask,
                "metadata": {
                    "total_objects": 0,
                    "classes_present": [],
                    "object_mapping": {},
                    "bbox_mapping": {}
                }
            }
        
        # 创建合并mask
        merged_mask = torch.zeros((self.mask_height, self.mask_width), dtype=torch.int32)
        
        # 构建元数据
        object_mapping = {}
        bbox_mapping = {}
        classes_present = set()
        
        for obj_id, obj_info in self.labels.items():
            # 将对象mask赋值到合并mask中
            merged_mask[obj_info.mask == True] = obj_id
            
            # 记录元数据
            object_mapping[obj_id] = obj_info.class_name
            bbox_mapping[obj_id] = [obj_info.x1, obj_info.y1, obj_info.x2, obj_info.y2]
            classes_present.add(obj_info.class_name)
        
        # 转换为numpy或保持tensor
        if return_type == "numpy":
            merged_mask = merged_mask.cpu().numpy()
        
        return {
            "merged_mask": merged_mask,
            "metadata": {
                "total_objects": len(self.labels),
                "classes_present": list(classes_present),
                "object_mapping": object_mapping,
                "bbox_mapping": bbox_mapping
            }
        }

    def get_binary_mask(self, class_names: List[str] = None) -> np.ndarray:
        """
        获取二值化mask（所有对象或指定类别）
        
        Args:
            class_names: 指定类别列表，None表示所有类别
        
        Returns:
            np.ndarray: (H, W) bool array
        """
        if not self.labels:
            return np.zeros((self.mask_height, self.mask_width), dtype=bool)
        
        binary_mask = torch.zeros((self.mask_height, self.mask_width), dtype=torch.bool)
        
        for obj_id, obj_info in self.labels.items():
            if class_names is None or obj_info.class_name in class_names:
                binary_mask = binary_mask | obj_info.mask
        
        return binary_mask.cpu().numpy()

    def get_flexible_mask_data(self, mode: str = "both") -> dict:
        """
        灵活的mask数据获取
        
        Args:
            mode: "class", "merged", "both"
        
        Returns:
            dict: 根据mode返回相应的数据
        """
        if mode == "class":
            return self.get_all_classes()
        elif mode == "merged":
            return self.get_merged_mask()
        elif mode == "both":
            return {
                "by_class": self.get_all_classes(),
                "merged": self.get_merged_mask()
            }
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'class', 'merged', or 'both'")

    def get_largest_object(self, class_name: str = None) -> Optional[dict]:
        """
        获取最大的对象（按面积计算）
        
        Args:
            class_name: 指定类别，None表示所有类别
        
        Returns:
            dict: 最大对象的信息，如果没有对象则返回None
        """
        if not self.labels:
            return None
        
        largest_obj = None
        max_area = 0
        
        for obj_id, obj_info in self.labels.items():
            if class_name is None or obj_info.class_name == class_name:
                area = obj_info.mask.sum().item()
                if area > max_area:
                    max_area = area
                    largest_obj = {
                        "instance_id": obj_id,
                        "class_name": obj_info.class_name,
                        "mask": obj_info.mask,
                        "bbox": [obj_info.x1, obj_info.y1, obj_info.x2, obj_info.y2],
                        "logit": obj_info.logit,
                        "area": area
                    }
        
        return largest_obj

    def get_objects_by_area_range(self, min_area: int = 0, max_area: int = None, 
                                 class_name: str = None) -> List[dict]:
        """
        根据面积范围获取对象
        
        Args:
            min_area: 最小面积
            max_area: 最大面积，None表示无上限
            class_name: 指定类别，None表示所有类别
        
        Returns:
            List[dict]: 符合条件的对象列表
        """
        if not self.labels:
            return []
        
        filtered_objects = []
        
        for obj_id, obj_info in self.labels.items():
            if class_name is None or obj_info.class_name == class_name:
                area = obj_info.mask.sum().item()
                
                if area >= min_area and (max_area is None or area <= max_area):
                    filtered_objects.append({
                        "instance_id": obj_id,
                        "class_name": obj_info.class_name,
                        "mask": obj_info.mask,
                        "bbox": [obj_info.x1, obj_info.y1, obj_info.x2, obj_info.y2],
                        "logit": obj_info.logit,
                        "area": area
                    })
        
        return filtered_objects


# 使用示例函数
def demonstrate_mask_usage():
    """
    演示如何使用扩展的MaskDictionaryModel
    """
    # 创建示例数据
    mask_dict = ExtendedMaskDictionaryModel()
    mask_dict.mask_height = 1080
    mask_dict.mask_width = 1920
    
    # 模拟添加一些对象
    # 这里只是示例，实际使用时对象会从检测结果中获取
    print("=== Mask数据使用演示 ===")
    
    # 1. 按类别获取对象
    print("\n1. 按类别获取对象:")
    hand_objects = mask_dict.get_masks_by_class("hand")
    print(f"手部对象: {len(hand_objects['objects'])} 个")
    
    # 2. 获取所有类别
    print("\n2. 获取所有类别:")
    all_classes = mask_dict.get_all_classes()
    for class_name, class_data in all_classes.items():
        print(f"  {class_name}: {len(class_data['objects'])} 个对象")
    
    # 3. 获取合并mask
    print("\n3. 获取合并mask:")
    merged_data = mask_dict.get_merged_mask()
    print(f"  总对象数: {merged_data['metadata']['total_objects']}")
    print(f"  类别: {merged_data['metadata']['classes_present']}")
    
    # 4. 获取二值化mask
    print("\n4. 获取二值化mask:")
    binary_mask = mask_dict.get_binary_mask()
    print(f"  二值化mask形状: {binary_mask.shape}")
    print(f"  非零像素数: {np.sum(binary_mask)}")
    
    # 5. 灵活数据获取
    print("\n5. 灵活数据获取:")
    flexible_data = mask_dict.get_flexible_mask_data("both")
    print(f"  包含类别数据: {'by_class' in flexible_data}")
    print(f"  包含合并数据: {'merged' in flexible_data}")


if __name__ == "__main__":
    demonstrate_mask_usage() 