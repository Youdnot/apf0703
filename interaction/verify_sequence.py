"""CPT序列验证脚本

用于验证生成的CPT序列的正确性和统计特征。
可以加载序列文件并进行详细分析，包括可视化播放模拟。

使用方法:
    python verify_sequence.py cpt_sequence.json
"""

import json
import time
import argparse
import statistics
from typing import Dict, List, Any
import logging


class CPTSequenceVerifier:
    """CPT序列验证器
    
    提供序列验证、统计分析和播放模拟功能。
    """
    
    def __init__(self):
        """初始化验证器"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_and_verify(self, filename: str) -> Dict[str, Any]:
        """加载并验证序列文件
        
        Args:
            filename: 序列文件名
            
        Returns:
            验证结果字典
        """
        try:
            # 加载序列
            with open(filename, 'r', encoding='utf-8') as f:
                sequence = json.load(f)
            
            self.logger.info(f"成功加载序列文件: {filename}")
            
            # 执行验证
            verification_results = self._verify_sequence(sequence)
            
            return {
                'sequence': sequence,
                'verification': verification_results,
                'valid': all(verification_results.values())
            }
            
        except FileNotFoundError:
            self.logger.error(f"文件未找到: {filename}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {e}")
            raise
    
    def _verify_sequence(self, sequence: Dict[str, Any]) -> Dict[str, bool]:
        """验证序列的各项指标
        
        Args:
            sequence: 序列字典
            
        Returns:
            各项验证结果
        """
        digits = sequence['digits']
        is_target = sequence['is_target']
        intervals = sequence['intervals']
        metadata = sequence.get('metadata', {})
        
        return {
            'length_consistency': self._verify_length_consistency(
                digits, is_target, intervals),
            'target_ratio': self._verify_target_ratio(
                is_target, metadata),
            'consecutive_targets': self._verify_consecutive_targets(
                is_target),
            'interval_range': self._verify_interval_range(
                intervals, metadata),
            'consecutive_intervals': self._verify_consecutive_intervals(
                intervals),
            'total_duration': self._verify_total_duration(
                digits, intervals, metadata)
        }
    
    def _verify_length_consistency(self, digits: List[int], 
                                  is_target: List[bool], 
                                  intervals: List[int]) -> bool:
        """验证各列表长度一致性"""
        lengths = [len(digits), len(is_target), len(intervals)]
        return len(set(lengths)) == 1
    
    def _verify_target_ratio(self, is_target: List[bool], 
                           metadata: Dict[str, Any]) -> bool:
        """验证目标比例是否符合预期"""
        if not metadata:
            return True
        
        actual_ratio = sum(is_target) / len(is_target)
        expected_ratio = metadata.get('config', {}).get('target_ratio', 0.3)
        
        # 允许±5%的误差
        tolerance = 0.05
        return abs(actual_ratio - expected_ratio) <= tolerance
    
    def _verify_consecutive_targets(self, is_target: List[bool], 
                                   max_consecutive: int = 3) -> bool:
        """验证连续目标数量限制"""
        consecutive_count = 0
        max_found = 0
        
        for target in is_target:
            if target:
                consecutive_count += 1
                max_found = max(max_found, consecutive_count)
            else:
                consecutive_count = 0
        
        return max_found <= max_consecutive
    
    def _verify_interval_range(self, intervals: List[int], 
                              metadata: Dict[str, Any]) -> bool:
        """验证间隔范围"""
        if not metadata:
            return True
        
        isi_range = metadata.get('config', {}).get('isi_range_ms', [500, 1000])
        min_isi, max_isi = isi_range
        
        return all(min_isi <= interval <= max_isi for interval in intervals)
    
    def _verify_consecutive_intervals(self, intervals: List[int], 
                                    max_consecutive: int = 3) -> bool:
        """验证连续相同间隔限制"""
        if len(intervals) <= 1:
            return True
        
        consecutive_count = 1
        for i in range(1, len(intervals)):
            if intervals[i] == intervals[i-1]:
                consecutive_count += 1
                if consecutive_count > max_consecutive:
                    return False
            else:
                consecutive_count = 1
        
        return True
    
    def _verify_total_duration(self, digits: List[int], intervals: List[int], 
                              metadata: Dict[str, Any]) -> bool:
        """验证总时长是否合理"""
        if not metadata:
            return True
        
        stimulus_duration = metadata.get('config', {}).get(
            'stimulus_duration_ms', 800)
        
        total_ms = len(digits) * stimulus_duration + sum(intervals)
        total_seconds = total_ms / 1000.0
        
        # 检查是否在合理范围内（45-75秒）
        return 45 <= total_seconds <= 75
    
    def print_detailed_analysis(self, verification_result: Dict[str, Any]) -> None:
        """打印详细的序列分析报告
        
        Args:
            verification_result: 验证结果字典
        """
        sequence = verification_result['sequence']
        verification = verification_result['verification']
        
        print("=" * 60)
        print("CPT序列详细分析报告")
        print("=" * 60)
        
        # 基本信息
        digits = sequence['digits']
        is_target = sequence['is_target']
        intervals = sequence['intervals']
        metadata = sequence.get('metadata', {})
        
        print(f"\n【基本信息】")
        print(f"序列长度: {len(digits)}")
        print(f"目标数字: {metadata.get('config', {}).get('target_digit', 'N/A')}")
        print(f"目标数量: {sum(is_target)}")
        print(f"非目标数量: {len(is_target) - sum(is_target)}")
        print(f"实际目标比例: {sum(is_target)/len(is_target):.1%}")
        
        # 时间统计
        if metadata:
            config = metadata.get('config', {})
            stimulus_duration = config.get('stimulus_duration_ms', 800)
            total_stimulus_time = len(digits) * stimulus_duration
            total_interval_time = sum(intervals)
            total_time = (total_stimulus_time + total_interval_time) / 1000.0
            
            print(f"\n【时间统计】")
            print(f"刺激总时间: {total_stimulus_time/1000.0:.1f}秒")
            print(f"间隔总时间: {total_interval_time/1000.0:.1f}秒")
            print(f"序列总时长: {total_time:.1f}秒")
            print(f"平均间隔: {statistics.mean(intervals):.0f}ms")
            print(f"间隔范围: {min(intervals)}-{max(intervals)}ms")
        
        # 连续性分析
        consecutive_targets = self._analyze_consecutive_targets(is_target)
        consecutive_intervals = self._analyze_consecutive_intervals(intervals)
        
        print(f"\n【连续性分析】")
        print(f"最大连续目标: {consecutive_targets['max_consecutive']}")
        print(f"连续目标序列: {consecutive_targets['sequences']}")
        print(f"最大连续相同间隔: {consecutive_intervals['max_consecutive']}")
        print(f"连续间隔详情: {consecutive_intervals['details']}")
        
        # 验证结果
        print(f"\n【验证结果】")
        for check, passed in verification.items():
            status = "✓ 通过" if passed else "✗ 失败"
            print(f"{check}: {status}")
        
        overall = "✓ 序列有效" if verification_result['valid'] else "✗ 序列无效"
        print(f"\n整体评估: {overall}")
    
    def _analyze_consecutive_targets(self, is_target: List[bool]) -> Dict[str, Any]:
        """分析连续目标情况"""
        sequences = []
        current_length = 0
        max_consecutive = 0
        
        for i, target in enumerate(is_target):
            if target:
                current_length += 1
                max_consecutive = max(max_consecutive, current_length)
            else:
                if current_length > 1:
                    sequences.append(f"位置{i-current_length}-{i-1}: {current_length}个连续目标")
                current_length = 0
        
        # 检查序列末尾
        if current_length > 1:
            sequences.append(f"位置{len(is_target)-current_length}-{len(is_target)-1}: {current_length}个连续目标")
        
        return {
            'max_consecutive': max_consecutive,
            'sequences': sequences if sequences else ['无连续目标序列']
        }
    
    def _analyze_consecutive_intervals(self, intervals: List[int]) -> Dict[str, Any]:
        """分析连续间隔情况"""
        if len(intervals) <= 1:
            return {'max_consecutive': 0, 'details': ['序列太短']}
        
        details = []
        current_value = intervals[0]
        current_length = 1
        max_consecutive = 1
        start_pos = 0
        
        for i in range(1, len(intervals)):
            if intervals[i] == current_value:
                current_length += 1
                max_consecutive = max(max_consecutive, current_length)
            else:
                if current_length > 2:
                    details.append(f"位置{start_pos}-{i-1}: {current_length}个{current_value}ms")
                current_value = intervals[i]
                current_length = 1
                start_pos = i
        
        # 检查序列末尾
        if current_length > 2:
            details.append(f"位置{start_pos}-{len(intervals)-1}: {current_length}个{current_value}ms")
        
        return {
            'max_consecutive': max_consecutive,
            'details': details if details else ['无过长连续间隔']
        }
    
    def simulate_playback(self, sequence: Dict[str, Any], 
                         speed_factor: float = 10.0) -> None:
        """模拟序列播放
        
        Args:
            sequence: 序列字典
            speed_factor: 播放速度倍数（加速）
        """
        digits = sequence['digits']
        is_target = sequence['is_target']
        intervals = sequence['intervals']
        metadata = sequence.get('metadata', {})
        
        stimulus_duration = metadata.get('config', {}).get(
            'stimulus_duration_ms', 800) / 1000.0 / speed_factor
        
        print(f"\n开始模拟播放 (速度x{speed_factor})...")
        print("按 Ctrl+C 停止播放\n")
        
        try:
            for i, (digit, target, interval) in enumerate(zip(digits, is_target, intervals)):
                target_str = "【目标】" if target else "【非目标】"
                print(f"第{i+1:2d}个刺激: {digit} {target_str}")
                
                # 模拟刺激显示时间
                time.sleep(stimulus_duration)
                
                # 模拟间隔时间
                interval_duration = interval / 1000.0 / speed_factor
                time.sleep(interval_duration)
                
        except KeyboardInterrupt:
            print("\n播放已停止")
        
        print("\n播放完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CPT序列验证工具')
    parser.add_argument('filename', help='序列文件名')
    parser.add_argument('--simulate', '-s', action='store_true', 
                       help='模拟播放序列')
    parser.add_argument('--speed', '-sp', type=float, default=10.0,
                       help='播放速度倍数 (默认: 10.0)')
    
    args = parser.parse_args()
    
    verifier = CPTSequenceVerifier()
    
    try:
        # 验证序列
        result = verifier.load_and_verify(args.filename)
        verifier.print_detailed_analysis(result)
        
        # 模拟播放
        if args.simulate:
            verifier.simulate_playback(result['sequence'], args.speed)
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# python verify_sequence.py cpt_sequence.json --simulate --speed=1