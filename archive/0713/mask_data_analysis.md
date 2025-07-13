# Mask数据结构和返回方式分析

## 1. current_mask_dict数据结构分析

### 1.1 基本结构
`current_mask_dict = tracker.last_mask_dict` 是一个 `MaskDictionaryModel` 实例，包含以下核心属性：

```python
@dataclass
class MaskDictionaryModel:
    mask_name: str = ""                    # 掩码文件名
    mask_height: int = 1080               # 掩码高度
    mask_width: int = 1920                # 掩码宽度
    promote_type: str = "mask"            # 提升类型
    labels: dict = field(default_factory=dict)  # 对象信息字典
```

### 1.2 labels字典结构
`labels` 是一个字典，键为实例ID，值为 `ObjectInfo` 对象：

```python
@dataclass
class ObjectInfo:
    instance_id: int = 0                  # 实例ID
    mask: any = None                      # 掩码数据 (torch.Tensor)
    class_name: str = ""                  # 类别名称
    x1, y1, x2, y2: int = 0              # 边界框坐标
    logit: float = 0.0                    # 置信度分数
```

### 1.3 实际数据示例
```python
current_mask_dict = {
    "mask_name": "mask_00001.npy",
    "mask_height": 1080,
    "mask_width": 1920,
    "promote_type": "mask",
    "labels": {
        1: ObjectInfo(
            instance_id=1,
            mask=torch.Tensor([[True, False, ...], ...]),  # (H, W) bool tensor
            class_name="hand",
            x1=100, y1=200, x2=300, y2=400,
            logit=0.95
        ),
        2: ObjectInfo(
            instance_id=2,
            mask=torch.Tensor([[False, True, ...], ...]),
            class_name="person",
            x1=400, y1=300, x2=600, y2=500,
            logit=0.87
        )
    }
}
```

## 2. 两种返回方式分析

### 2.1 方式一：按类别分别返回

#### 2.1.1 实现方案
```python
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
            "hand": {objects: [...]},
            "person": {objects: [...]},
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
```

#### 2.1.2 使用示例
```python
# 获取特定类别的对象
hand_objects = current_mask_dict.get_masks_by_class("hand")
if hand_objects["objects"]:
    print(f"检测到 {len(hand_objects['objects'])} 个手部对象")
    for obj in hand_objects["objects"]:
        print(f"  实例ID: {obj['instance_id']}, 面积: {obj['area']}")

# 获取所有类别
all_classes = current_mask_dict.get_all_classes()
for class_name, class_data in all_classes.items():
    print(f"类别 {class_name}: {len(class_data['objects'])} 个对象")
```

#### 2.1.3 优点
- 便于按类别处理对象
- 支持类别特定的后处理
- 内存效率高（只返回需要的类别）
- 便于下游应用分类处理

#### 2.1.4 缺点
- 需要额外的分类逻辑
- 可能丢失类别间的关系信息

### 2.2 方式二：整体mask合并后返回

#### 2.2.1 实现方案
```python
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

def get_binary_mask(self, class_names: list = None) -> np.ndarray:
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
```

#### 2.2.2 使用示例
```python
# 获取合并mask
merged_data = current_mask_dict.get_merged_mask()
merged_mask = merged_data["merged_mask"]  # (H, W) int32
metadata = merged_data["metadata"]

print(f"总对象数: {metadata['total_objects']}")
print(f"类别: {metadata['classes_present']}")

# 获取二值化mask（所有对象）
binary_mask = current_mask_dict.get_binary_mask()

# 获取特定类别的二值化mask
hand_binary_mask = current_mask_dict.get_binary_mask(["hand"])
```

#### 2.2.3 优点
- 统一的mask格式，便于处理
- 保留所有对象信息
- 适合需要整体mask的应用场景
- 便于可视化和存储

#### 2.2.4 缺点
- 可能包含不需要的对象
- 内存占用较大
- 需要额外的后处理来提取特定类别

## 3. 推荐的使用策略

### 3.1 根据应用场景选择

#### 场景1：需要按类别处理
```python
# 推荐使用方式一
hand_objects = current_mask_dict.get_masks_by_class("hand")
for obj in hand_objects["objects"]:
    # 处理手部对象
    process_hand_mask(obj["mask"])
```

#### 场景2：需要整体mask
```python
# 推荐使用方式二
merged_data = current_mask_dict.get_merged_mask()
# 用于障碍物检测、路径规划等
use_merged_mask_for_path_planning(merged_data["merged_mask"])
```

### 3.2 混合使用策略
```python
def get_flexible_mask_data(self, mode: str = "both"):
    """
    灵活的mask数据获取
    
    Args:
        mode: "class", "merged", "both"
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
```

## 4. 性能考虑

### 4.1 内存使用
- **方式一**：按需分配，内存效率高
- **方式二**：需要完整mask，内存占用较大

### 4.2 计算复杂度
- **方式一**：O(n)，n为对象数量
- **方式二**：O(n × H × W)，需要遍历整个mask

### 4.3 推荐使用场景
- **方式一**：实时处理、类别特定应用
- **方式二**：离线分析、整体mask应用 