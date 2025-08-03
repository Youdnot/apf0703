"""CPT数字序列生成器

该模块实现持续表现任务(Continuous Performance Task)的数字序列生成功能。
生成标准化的刺激序列，包含目标和非目标数字，以及对应的刺激间隔。

使用方法:
    generator = CPTSequenceGenerator(target_digit=5, target_ratio=0.3)
    sequence = generator.generate_sequence()
    generator.save_sequence(sequence, 'cpt_sequence.json')
"""

import random
import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class CPTConfig:
    """CPT任务配置参数
    
    Attributes:
        digits_range: 可用数字范围 (start, end)
        target_digit: 目标数字
        sequence_length: 序列长度
        target_ratio: 目标数字比例
        stimulus_duration: 刺激持续时间(ms)
        isi_range: 刺激间隔范围(ms) (min, max)
        max_consecutive_targets: 最大连续目标数量
        max_consecutive_intervals: 最大连续相同间隔数量
    """
    digits_range: Tuple[int, int] = (1, 9)
    target_digit: int = 5
    sequence_length: int = 40
    target_ratio: float = 0.3
    stimulus_duration: int = 800
    isi_range: Tuple[int, int] = (500, 1000)
    max_consecutive_targets: int = 3
    max_consecutive_intervals: int = 3


class CPTSequenceGenerator:
    """CPT序列生成器
    
    生成符合CPT标准的数字序列，包含以下功能：
    - 生成指定比例的目标和非目标数字
    - 随机打乱序列顺序
    - 验证连续目标数量限制
    - 生成刺激间隔时间
    - 验证总时长和间隔异常
    """
    
    def __init__(self, config: CPTConfig = None):
        """初始化序列生成器
        
        Args:
            config: CPT配置参数，如果为None则使用默认配置
        """
        self.config = config or CPTConfig()
        self.digits = list(range(self.config.digits_range[0], 
                                self.config.digits_range[1] + 1))
        self.non_target_digits = [d for d in self.digits 
                                 if d != self.config.target_digit]
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self) -> None:
        """验证配置参数的有效性
        
        Raises:
            ValueError: 当配置参数无效时
        """
        if self.config.target_digit not in self.digits:
            raise ValueError(f"目标数字 {self.config.target_digit} 不在数字范围内")
        
        if not 0 < self.config.target_ratio < 1:
            raise ValueError(f"目标比例 {self.config.target_ratio} 必须在0-1之间")
        
        if self.config.sequence_length <= 0:
            raise ValueError(f"序列长度 {self.config.sequence_length} 必须大于0")
        
        expected_targets = int(self.config.sequence_length * self.config.target_ratio)
        if expected_targets == 0:
            raise ValueError("根据当前设置，目标数字数量为0")
    
    def generate_sequence(self, max_attempts: int = 100) -> Dict[str, List[Any]]:
        """生成CPT序列
        
        生成包含数字、目标标识和刺激间隔的完整序列。
        如果生成的序列不符合连续性限制，会重新生成。
        
        Args:
            max_attempts: 最大重试次数
            
        Returns:
            包含以下键的字典：
            - 'digits': 显示的数字列表
            - 'is_target': 是否为目标的布尔值列表
            - 'intervals': 刺激间隔列表(ms)
            - 'metadata': 序列元数据
            
        Raises:
            RuntimeError: 超过最大重试次数仍无法生成有效序列
        """
        for attempt in range(max_attempts):
            self.logger.info(f"尝试生成序列 (第 {attempt + 1} 次)")
            
            # 1. 生成基础序列
            digits, is_target = self._generate_basic_sequence()
            
            # 2. 随机打乱
            digits, is_target = self._shuffle_sequence(digits, is_target)
            
            # 3. 验证连续目标
            if not self._validate_consecutive_targets(is_target):
                self.logger.warning(f"第 {attempt + 1} 次尝试：连续目标超过限制")
                continue
            
            # 4. 生成刺激间隔
            intervals = self._generate_intervals()
            
            # 5. 验证间隔异常
            if not self._validate_intervals(intervals):
                self.logger.warning(f"第 {attempt + 1} 次尝试：间隔验证失败")
                continue
            
            # 6. 生成元数据
            metadata = self._generate_metadata(digits, is_target, intervals)
            
            self.logger.info(f"序列生成成功 (第 {attempt + 1} 次尝试)")
            return {
                'digits': digits,
                'is_target': is_target,
                'intervals': intervals,
                'metadata': metadata
            }
        
        raise RuntimeError(f"超过最大重试次数 ({max_attempts})，无法生成有效序列")
    
    def _generate_basic_sequence(self) -> Tuple[List[int], List[bool]]:
        """生成基础序列（未打乱）
        
        Returns:
            (数字列表, 目标标识列表)
        """
        target_count = int(self.config.sequence_length * self.config.target_ratio)
        non_target_count = self.config.sequence_length - target_count
        
        # 生成目标数字
        digits = [self.config.target_digit] * target_count
        is_target = [True] * target_count
        
        # 生成非目标数字
        for _ in range(non_target_count):
            non_target = random.choice(self.non_target_digits)
            digits.append(non_target)
            is_target.append(False)
        
        return digits, is_target
    
    def _shuffle_sequence(self, digits: List[int], 
                         is_target: List[bool]) -> Tuple[List[int], List[bool]]:
        """随机打乱序列
        
        Args:
            digits: 数字列表
            is_target: 目标标识列表
            
        Returns:
            (打乱后的数字列表, 打乱后的目标标识列表)
        """
        combined = list(zip(digits, is_target))
        random.shuffle(combined)
        shuffled_digits, shuffled_is_target = zip(*combined)
        return list(shuffled_digits), list(shuffled_is_target)
    
    def _validate_consecutive_targets(self, is_target: List[bool]) -> bool:
        """验证连续目标数量是否超过限制
        
        Args:
            is_target: 目标标识列表
            
        Returns:
            True如果没有超过连续限制，False否则
        """
        consecutive_count = 0
        max_consecutive = 0
        
        for target in is_target:
            if target:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        return max_consecutive <= self.config.max_consecutive_targets
    
    def _generate_intervals(self) -> List[int]:
        """生成刺激间隔序列
        
        在指定范围内生成随机间隔，使用100ms步长。
        
        Returns:
            间隔时间列表(ms)
        """
        min_isi, max_isi = self.config.isi_range
        possible_intervals = list(range(min_isi, max_isi + 1, 100))
        
        intervals = []
        for _ in range(self.config.sequence_length):
            interval = random.choice(possible_intervals)
            intervals.append(interval)
        
        return intervals
    
    def _validate_intervals(self, intervals: List[int]) -> bool:
        """验证间隔序列是否有异常
        
        检查：
        1. 连续相同间隔数量
        2. 间隔值是否在有效范围内
        
        Args:
            intervals: 间隔列表
            
        Returns:
            True如果间隔有效，False否则
        """
        # 检查连续相同间隔
        consecutive_count = 1
        for i in range(1, len(intervals)):
            if intervals[i] == intervals[i-1]:
                consecutive_count += 1
                if consecutive_count > self.config.max_consecutive_intervals:
                    return False
            else:
                consecutive_count = 1
        
        # 检查间隔范围
        min_isi, max_isi = self.config.isi_range
        for interval in intervals:
            if not (min_isi <= interval <= max_isi):
                return False
        
        return True
    
    def _generate_metadata(self, digits: List[int], is_target: List[bool], 
                          intervals: List[int]) -> Dict[str, Any]:
        """生成序列元数据
        
        Args:
            digits: 数字列表
            is_target: 目标标识列表
            intervals: 间隔列表
            
        Returns:
            包含序列统计信息的字典
        """
        target_count = sum(is_target)
        total_time = (self.config.stimulus_duration * len(digits) + 
                     sum(intervals)) / 1000.0  # 转换为秒
        
        return {
            'config': {
                'target_digit': self.config.target_digit,
                'sequence_length': self.config.sequence_length,
                'target_ratio': self.config.target_ratio,
                'stimulus_duration_ms': self.config.stimulus_duration,
                'isi_range_ms': self.config.isi_range
            },
            'statistics': {
                'total_stimuli': len(digits),
                'target_count': target_count,
                'non_target_count': len(digits) - target_count,
                'actual_target_ratio': target_count / len(digits),
                'total_duration_seconds': total_time,
                'average_interval_ms': sum(intervals) / len(intervals)
            }
        }
    
    def save_sequence(self, sequence: Dict[str, Any], filename: str) -> None:
        """保存序列到文件
        
        Args:
            sequence: 生成的序列字典
            filename: 保存文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(sequence, f, indent=2, ensure_ascii=False)
            self.logger.info(f"序列已保存到: {filename}")
        except Exception as e:
            self.logger.error(f"保存序列失败: {e}")
            raise
    
    @staticmethod
    def load_sequence(filename: str) -> Dict[str, Any]:
        """从文件加载序列
        
        Args:
            filename: 文件名
            
        Returns:
            序列字典
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载序列失败: {e}")
            raise


def main():
    """主函数示例"""
    # 创建配置
    config = CPTConfig(
        target_digit=5,
        target_ratio=0.3,
        sequence_length=40,
        isi_range=(500, 1000)
    )
    
    # 创建生成器
    generator = CPTSequenceGenerator(config)
    
    # 生成序列
    sequence = generator.generate_sequence()
    
    # 保存序列
    generator.save_sequence(sequence, 'cpt_sequence.json')
    
    # 打印统计信息
    metadata = sequence['metadata']
    print("\n=== 序列生成完成 ===")
    print(f"序列长度: {metadata['statistics']['total_stimuli']}")
    print(f"目标数量: {metadata['statistics']['target_count']}")
    print(f"实际目标比例: {metadata['statistics']['actual_target_ratio']:.1%}")
    print(f"预计总时长: {metadata['statistics']['total_duration_seconds']:.1f}秒")


if __name__ == "__main__":
    main()
