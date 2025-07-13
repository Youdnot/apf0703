#!/usr/bin/env python3
"""
Mask输出管理器使用示例
展示如何在下游应用中使用实时检测的mask数据
"""

import numpy as np
import cv2
from typing import List, Optional
import time

# 假设这是从tracking_unit_test.py导入的MaskOutputManager
from tracking_unit_test import MaskOutputManager


class DownstreamApplication:
    """
    下游应用示例类
    展示如何使用MaskOutputManager进行各种应用
    """
    
    def __init__(self):
        self.mask_manager = MaskOutputManager(max_cache_size=30)
        self.last_processed_frame = -1
        
    def process_mask_for_gesture_recognition(self) -> Optional[dict]:
        """
        基于mask进行手势识别
        """
        current_mask = self.mask_manager.get_current_mask()
        if current_mask is None:
            return None
            
        # 获取手部对象
        hand_objects = self.mask_manager.get_objects_by_class("hand")
        if not hand_objects:
            return None
            
        # 分析手部mask进行手势识别
        gesture_info = []
        for hand_obj in hand_objects:
            hand_mask = hand_obj['mask']
            bbox = hand_obj['bbox']
            
            # 简单的面积和位置分析
            area = np.sum(hand_mask)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # 基于面积判断手势类型（示例）
            if area > 50000:
                gesture_type = "open_hand"
            elif area > 30000:
                gesture_type = "closed_fist"
            else:
                gesture_type = "pointing"
                
            gesture_info.append({
                'instance_id': hand_obj['instance_id'],
                'gesture_type': gesture_type,
                'area': area,
                'center': (center_x, center_y),
                'bbox': bbox
            })
            
        return {
            'frame_idx': self.mask_manager.frame_count,
            'gestures': gesture_info
        }
    
    def track_object_movement(self, class_name: str = None) -> Optional[dict]:
        """
        跟踪对象运动
        """
        history = self.mask_manager.get_mask_history(num_frames=5)
        if len(history) < 2:
            return None
            
        # 获取最大对象的历史轨迹
        trajectories = []
        for i, frame_data in enumerate(history):
            largest_obj = self._get_largest_object_from_frame(frame_data, class_name)
            if largest_obj:
                trajectories.append({
                    'frame_idx': frame_data['frame_idx'],
                    'timestamp': frame_data['timestamp'],
                    'center': largest_obj['center'],
                    'area': largest_obj['area']
                })
                
        if len(trajectories) < 2:
            return None
            
        # 计算运动信息
        movement_info = {
            'start_frame': trajectories[0]['frame_idx'],
            'end_frame': trajectories[-1]['frame_idx'],
            'total_frames': len(trajectories),
            'start_center': trajectories[0]['center'],
            'end_center': trajectories[-1]['center'],
            'movement_vector': (
                trajectories[-1]['center'][0] - trajectories[0]['center'][0],
                trajectories[-1]['center'][1] - trajectories[0]['center'][1]
            ),
            'area_change': trajectories[-1]['area'] - trajectories[0]['area']
        }
        
        return movement_info
    
    def _get_largest_object_from_frame(self, frame_data: dict, class_name: str = None) -> Optional[dict]:
        """
        从单帧数据中获取最大对象
        """
        mask = frame_data['mask']
        metadata = frame_data['metadata']
        labels = metadata.get('labels', {})
        
        largest_obj = None
        max_area = 0
        
        for obj_id, obj_info in labels.items():
            if class_name and obj_info.get('class_name', '').lower() != class_name.lower():
                continue
                
            instance_id = obj_info.get('instance_id', 0)
            if instance_id == 0:
                continue
                
            obj_mask = mask == instance_id
            area = np.sum(obj_mask)
            
            if area > max_area:
                max_area = area
                bbox = [obj_info.get('x1', 0), obj_info.get('y1', 0), 
                       obj_info.get('x2', 0), obj_info.get('y2', 0)]
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                largest_obj = {
                    'instance_id': instance_id,
                    'class_name': obj_info.get('class_name'),
                    'area': area,
                    'center': center,
                    'bbox': bbox
                }
        
        return largest_obj
    
    def generate_mask_visualization(self) -> Optional[np.ndarray]:
        """
        生成mask可视化图像
        """
        current_mask = self.mask_manager.get_current_mask()
        current_frame = self.mask_manager.get_current_frame()
        
        if current_mask is None or current_frame is None:
            return None
            
        # 创建彩色mask可视化
        H, W = current_mask.shape
        mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 为每个对象分配不同颜色
        unique_ids = np.unique(current_mask)
        colors = [
            (255, 0, 0),   # 红色
            (0, 255, 0),   # 绿色
            (0, 0, 255),   # 蓝色
            (255, 255, 0), # 黄色
            (255, 0, 255), # 紫色
            (0, 255, 255), # 青色
        ]
        
        for i, obj_id in enumerate(unique_ids):
            if obj_id == 0:  # 跳过背景
                continue
            color = colors[i % len(colors)]
            mask_vis[current_mask == obj_id] = color
            
        # 与原图混合
        alpha = 0.6
        result = cv2.addWeighted(current_frame, 1-alpha, mask_vis, alpha, 0)
        
        return result
    
    def export_analysis_report(self, output_path: str) -> None:
        """
        导出分析报告
        """
        current_metadata = self.mask_manager.get_current_metadata()
        if current_metadata is None:
            print("没有可用的元数据进行报告")
            return
            
        report = {
            'timestamp': time.time(),
            'frame_idx': self.mask_manager.frame_count,
            'total_objects': len(current_metadata.get('labels', {})),
            'object_classes': list(set([
                obj_info.get('class_name', 'unknown') 
                for obj_info in current_metadata.get('labels', {}).values()
            ])),
            'mask_shape': self.mask_manager.get_current_mask().shape if self.mask_manager.get_current_mask() is not None else None
        }
        
        # 保存报告
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"分析报告已导出到: {output_path}")


def example_usage():
    """
    使用示例
    """
    print("=== Mask输出管理器使用示例 ===")
    
    # 创建下游应用实例
    app = DownstreamApplication()
    
    # 模拟mask数据更新（在实际应用中，这些数据来自tracking_unit_test.py）
    print("1. 模拟mask数据更新...")
    
    # 这里应该从实际的tracker获取数据
    # 在实际使用中，这些调用会在tracking_unit_test.py的main函数中进行
    
    print("2. 手势识别示例:")
    gesture_result = app.process_mask_for_gesture_recognition()
    if gesture_result:
        print(f"   检测到手势: {gesture_result}")
    
    print("3. 对象运动跟踪示例:")
    movement_result = app.track_object_movement()
    if movement_result:
        print(f"   运动信息: {movement_result}")
    
    print("4. 生成可视化示例:")
    vis_result = app.generate_mask_visualization()
    if vis_result is not None:
        print(f"   可视化图像形状: {vis_result.shape}")
    
    print("5. 导出分析报告示例:")
    app.export_analysis_report("./outputs/analysis_report.json")
    
    print("=== 示例完成 ===")


if __name__ == "__main__":
    example_usage() 