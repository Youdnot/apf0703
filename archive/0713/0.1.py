import copy
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from external.grounding_sam2.sam2.build_sam import build_sam2, build_sam2_video_predictor
from external.grounding_sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from external.grounding_sam2.utils.common_utils import CommonUtils
from external.grounding_sam2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from external.grounding_sam2.utils.track_utils import sample_points_from_masks
from external.grounding_sam2.utils.video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


from utils.camera_tracking import *

import os

import cv2
import torch
from external.grounding_sam2.utils.common_utils import CommonUtils


class MaskOutputManager:
    """
    管理实时检测的mask输出，提供可扩展的接口供下游应用使用
    """
    
    def __init__(self, max_cache_size: int = 10):
        """
        初始化mask输出管理器
        
        Args:
            max_cache_size: 最大缓存帧数
        """
        self.max_cache_size = max_cache_size
        self.mask_cache = deque(maxlen=max_cache_size)
        self.metadata_cache = deque(maxlen=max_cache_size)
        self.frame_cache = deque(maxlen=max_cache_size)
        self.current_mask = None
        self.current_metadata = None
        self.current_frame = None
        self.frame_count = 0
        
    def update_mask(self, 
                   mask_array: np.ndarray, 
                   metadata: dict, 
                   frame: np.ndarray,
                   frame_idx: int) -> None:
        """
        更新当前的mask数据
        
        Args:
            mask_array: 检测到的mask数组 (H, W)
            metadata: 包含对象信息的元数据字典
            frame: 原始RGB图像帧
            frame_idx: 帧索引
        """
        self.current_mask = mask_array.copy()
        self.current_metadata = metadata.copy()
        self.current_frame = frame.copy()
        self.frame_count = frame_idx
        
        # 添加到缓存
        cache_entry = {
            'frame_idx': frame_idx,
            'timestamp': time.time(),
            'mask': mask_array.copy(),
            'metadata': metadata.copy(),
            'frame': frame.copy()
        }
        
        self.mask_cache.append(cache_entry)
        self.metadata_cache.append(metadata.copy())
        self.frame_cache.append(frame.copy())
        
        print(f"[MaskManager] 更新mask数据 - 帧 {frame_idx}, 对象数量: {len(metadata.get('labels', {}))}")
    
    def get_current_mask(self) -> Optional[np.ndarray]:
        """获取当前帧的mask"""
        return self.current_mask
    
    def get_current_metadata(self) -> Optional[dict]:
        """获取当前帧的元数据"""
        return self.current_metadata
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """获取当前帧"""
        return self.current_frame
    
    def get_mask_history(self, num_frames: int = 5) -> List[dict]:
        """
        获取历史mask数据
        
        Args:
            num_frames: 返回的帧数
            
        Returns:
            包含历史mask数据的列表
        """
        return list(self.mask_cache)[-num_frames:]
    
    def get_objects_by_class(self, class_name: str) -> List[dict]:
        """
        根据类别名称获取对象信息
        
        Args:
            class_name: 目标类别名称
            
        Returns:
            匹配的对象信息列表
        """
        if self.current_metadata is None:
            return []
        
        objects = []
        labels = self.current_metadata.get('labels', {})
        
        for obj_id, obj_info in labels.items():
            if obj_info.get('class_name', '').lower() == class_name.lower():
                objects.append({
                    'instance_id': obj_info.get('instance_id'),
                    'class_name': obj_info.get('class_name'),
                    'bbox': [obj_info.get('x1', 0), obj_info.get('y1', 0), 
                            obj_info.get('x2', 0), obj_info.get('y2', 0)],
                    'mask': self.current_mask == obj_info.get('instance_id', 0)
                })
        
        return objects
    
    def get_largest_object(self, class_name: str = None) -> Optional[dict]:
        """
        获取最大的对象（按mask面积计算）
        
        Args:
            class_name: 可选的类别过滤
            
        Returns:
            最大对象的信息字典
        """
        if self.current_metadata is None:
            return None
        
        labels = self.current_metadata.get('labels', {})
        largest_obj = None
        max_area = 0
        
        for obj_id, obj_info in labels.items():
            if class_name and obj_info.get('class_name', '').lower() != class_name.lower():
                continue
                
            instance_id = obj_info.get('instance_id', 0)
            if instance_id == 0:
                continue
                
            mask = self.current_mask == instance_id
            area = np.sum(mask)
            
            if area > max_area:
                max_area = area
                largest_obj = {
                    'instance_id': instance_id,
                    'class_name': obj_info.get('class_name'),
                    'bbox': [obj_info.get('x1', 0), obj_info.get('y1', 0), 
                            obj_info.get('x2', 0), obj_info.get('y2', 0)],
                    'mask': mask,
                    'area': area
                }
        
        return largest_obj
    
    def export_mask_data(self, output_path: str) -> None:
        """
        导出当前mask数据到文件
        
        Args:
            output_path: 输出文件路径
        """
        if self.current_mask is None:
            print("[MaskManager] 没有可导出的mask数据")
            return
            
        data = {
            'frame_idx': self.frame_count,
            'timestamp': time.time(),
            'mask': self.current_mask.tolist(),
            'metadata': self.current_metadata,
            'shape': self.current_mask.shape
        }
        
        np.save(output_path, data)
        print(f"[MaskManager] Mask数据已导出到: {output_path}")


def main():
    # Parameter settings
    output_dir = "./outputs"
    prompt_text = "hand."
    detection_interval = 20
    max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize the object tracker
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt("obstacle. person. viehicle. car. bus. truck. desk. table. chair.")

    # 初始化mask输出管理器
    mask_manager = MaskOutputManager(max_cache_size=20)

    # Open the camera (or replace with local video file, e.g., cv2.VideoCapture("video.mp4"))
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("assets/walking test data.mp4")
    if not cap.isOpened():
        print("[Error] Cannot open camera.")
        return

    print("[Info] Camera opened. Press 'q' to quit.")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"[Frame {frame_idx}] Processing live frame...")
            process_image = tracker.add_image(frame_rgb)

            if process_image is None or not isinstance(process_image, np.ndarray):
                print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
                frame_idx += 1
                continue

            # 从tracker获取当前mask和元数据
            current_mask_dict = tracker.last_mask_dict
            if current_mask_dict and hasattr(current_mask_dict, 'labels') and current_mask_dict.labels:
                # 构建mask数组
                H, W = frame_rgb.shape[:2]
                mask_img = torch.zeros((H, W), dtype=torch.int32)
                for obj_id, obj_info in current_mask_dict.labels.items():
                    mask_img[obj_info.mask == True] = obj_id
                mask_array = mask_img.cpu().numpy()
                
                # 更新mask管理器
                mask_manager.update_mask(
                    mask_array=mask_array,
                    metadata=current_mask_dict.to_dict(),
                    frame=frame_rgb,
                    frame_idx=frame_idx
                )
                
                # 示例：获取特定类别的对象
                hand_objects = mask_manager.get_objects_by_class("hand")
                if hand_objects:
                    print(f"[MaskManager] 检测到 {len(hand_objects)} 个手部对象")
                
                # 示例：获取最大对象
                largest_obj = mask_manager.get_largest_object()
                if largest_obj:
                    print(f"[MaskManager] 最大对象: {largest_obj['class_name']}, 面积: {largest_obj['area']}")
                
                # 示例：每10帧导出一次mask数据
                # if frame_idx % 10 == 0:
                #     export_path = os.path.join(output_dir, f"mask_export_{frame_idx:05d}.npy")
                #     mask_manager.export_mask_data(export_path)

            # process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Live Inference", process_image_bgr)

            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("[Info] Quit signal received.")
            #     break

            tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
            frame_idx += 1

            # if frame_idx >= max_frames:
            #     print(f"[Info] Reached max_frames {max_frames}. Stopping.")
            #     break
    except KeyboardInterrupt:
        print("[Info] Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Done] Live inference complete.")
        
        # 最终导出最后的mask数据
        # if mask_manager.get_current_mask() is not None:
        #     final_export_path = os.path.join(output_dir, "final_mask_export.npy")
        #     mask_manager.export_mask_data(final_export_path)

if __name__ == "__main__":
    main()