# Mask输出系统使用指南

## 概述

本系统基于Grounding-SAM2模型对摄像头输入进行连续物体检测，并提供实时的mask数据输出接口，便于下游应用使用。

## 主要功能

### 1. 实时物体检测
- 使用GroundingDINO进行零样本物体检测
- 使用SAM2进行精确的mask分割
- 支持连续帧的增量跟踪

### 2. Mask数据管理
- 实时mask数据缓存
- 历史数据查询
- 按类别过滤对象
- 对象运动跟踪

### 3. 可扩展的输出接口
- 获取当前mask数据
- 查询特定类别对象
- 导出mask数据到文件
- 生成可视化结果

## 文件结构

```
├── tracking_unit_test.py      # 主要的检测和mask输出程序
├── mask_usage_example.py      # 下游应用使用示例
├── utils/camera_tracking.py   # 核心检测类定义
└── MASK_OUTPUT_README.md      # 本说明文件
```

## 使用方法

### 1. 运行主检测程序

```bash
python tracking_unit_test.py
```

这将启动摄像头检测，并实时输出mask数据。

### 2. 在代码中使用MaskOutputManager

```python
from tracking_unit_test import MaskOutputManager

# 创建mask管理器
mask_manager = MaskOutputManager(max_cache_size=20)

# 获取当前mask
current_mask = mask_manager.get_current_mask()

# 获取特定类别的对象
hand_objects = mask_manager.get_objects_by_class("hand")

# 获取最大对象
largest_obj = mask_manager.get_largest_object()

# 获取历史数据
history = mask_manager.get_mask_history(num_frames=5)
```

## MaskOutputManager API

### 初始化
```python
mask_manager = MaskOutputManager(max_cache_size=10)
```

### 主要方法

#### 1. 数据更新
```python
mask_manager.update_mask(
    mask_array=mask_array,      # numpy数组 (H, W)
    metadata=metadata_dict,     # 包含对象信息的字典
    frame=frame_rgb,           # 原始RGB图像
    frame_idx=frame_idx        # 帧索引
)
```

#### 2. 数据获取
```python
# 获取当前mask
current_mask = mask_manager.get_current_mask()

# 获取当前元数据
metadata = mask_manager.get_current_metadata()

# 获取当前帧
frame = mask_manager.get_current_frame()
```

#### 3. 对象查询
```python
# 按类别获取对象
objects = mask_manager.get_objects_by_class("hand")

# 获取最大对象
largest = mask_manager.get_largest_object()

# 获取特定类别的最大对象
largest_hand = mask_manager.get_largest_object("hand")
```

#### 4. 历史数据
```python
# 获取最近5帧的历史数据
history = mask_manager.get_mask_history(num_frames=5)
```

#### 5. 数据导出
```python
# 导出mask数据到文件
mask_manager.export_mask_data("output.npy")
```

## 下游应用示例

### 1. 手势识别
```python
def process_gesture(mask_manager):
    hand_objects = mask_manager.get_objects_by_class("hand")
    for hand in hand_objects:
        area = np.sum(hand['mask'])
        if area > 50000:
            return "open_hand"
        elif area > 30000:
            return "closed_fist"
        else:
            return "pointing"
```

### 2. 对象运动跟踪
```python
def track_movement(mask_manager):
    history = mask_manager.get_mask_history(5)
    if len(history) >= 2:
        start_pos = history[0]['center']
        end_pos = history[-1]['center']
        movement = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        return movement
```

### 3. 实时可视化
```python
def visualize_mask(mask_manager):
    mask = mask_manager.get_current_mask()
    frame = mask_manager.get_current_frame()
    
    # 创建彩色mask
    mask_vis = np.zeros_like(frame)
    unique_ids = np.unique(mask)
    
    for obj_id in unique_ids:
        if obj_id == 0:  # 跳过背景
            continue
        color = np.random.randint(0, 255, 3)
        mask_vis[mask == obj_id] = color
    
    # 与原图混合
    result = cv2.addWeighted(frame, 0.7, mask_vis, 0.3, 0)
    return result
```

## 数据格式

### Mask数组格式
- 形状: `(H, W)`
- 数据类型: `numpy.ndarray`
- 值含义: 0表示背景，正整数表示对象ID

### 元数据格式
```python
{
    "labels": {
        "obj_1": {
            "instance_id": 1,
            "class_name": "hand",
            "x1": 100, "y1": 200, "x2": 300, "y2": 400,
            "mask": numpy.ndarray  # 布尔mask
        }
    },
    "mask_name": "mask_00001.npy",
    "mask_height": 480,
    "mask_width": 640
}
```

## 配置参数

### 检测参数
- `prompt_text`: 检测提示词
- `detection_interval`: 检测间隔帧数
- `max_frames`: 最大处理帧数

### 缓存参数
- `max_cache_size`: 最大缓存帧数

## 性能优化

1. **内存管理**: 定期清理缓存避免内存溢出
2. **GPU优化**: 使用CUDA加速计算
3. **批处理**: 支持批量处理多帧数据

## 常见问题

### Q: 如何修改检测的物体类别？
A: 修改 `tracker.set_prompt()` 中的提示词，例如：
```python
tracker.set_prompt("person. car. hand. table.")
```

### Q: 如何调整检测频率？
A: 修改 `detection_interval` 参数：
```python
tracker = IncrementalObjectTracker(detection_interval=10)  # 每10帧检测一次
```

### Q: 如何获取特定对象的mask？
A: 使用对象ID过滤：
```python
current_mask = mask_manager.get_current_mask()
object_mask = current_mask == object_id
```

## 扩展开发

### 添加新的下游应用
1. 继承或使用 `MaskOutputManager`
2. 实现特定的处理逻辑
3. 集成到主检测循环中

### 自定义数据格式
1. 修改 `update_mask()` 方法
2. 添加新的数据字段
3. 实现相应的获取方法

## 注意事项

1. 确保有足够的GPU内存运行模型
2. 定期清理缓存避免内存泄漏
3. 根据实际需求调整检测间隔
4. 注意mask数据的坐标系一致性 