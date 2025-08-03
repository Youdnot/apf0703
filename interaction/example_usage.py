"""CPT序列生成器使用示例

演示如何使用CPT序列生成器创建不同配置的序列。
包含多种常用场景的示例代码。

运行方法:
    python example_usage.py
"""

from seq import CPTSequenceGenerator, CPTConfig


def example_basic_usage():
    """基础使用示例"""
    print("=" * 50)
    print("示例1: 基础使用")
    print("=" * 50)
    
    # 使用默认配置
    generator = CPTSequenceGenerator()
    sequence = generator.generate_sequence()
    
    # 保存序列
    filename = 'basic_sequence.json'
    generator.save_sequence(sequence, filename)
    
    # 显示统计信息
    stats = sequence['metadata']['statistics']
    print(f"生成序列长度: {stats['total_stimuli']}")
    print(f"目标数量: {stats['target_count']}")
    print(f"预计时长: {stats['total_duration_seconds']:.1f}秒")
    print(f"文件已保存: {filename}")


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 50)
    print("示例2: 自定义配置")
    print("=" * 50)
    
    # 创建自定义配置
    config = CPTConfig(
        target_digit=7,          # 目标数字改为7
        target_ratio=0.25,       # 目标比例25%
        sequence_length=50,      # 序列长度50
        isi_range=(600, 1200),   # 间隔范围600-1200ms
        max_consecutive_targets=2  # 最大连续目标数减少到2
    )
    
    generator = CPTSequenceGenerator(config)
    sequence = generator.generate_sequence()
    
    # 保存序列
    filename = 'custom_sequence.json'
    generator.save_sequence(sequence, filename)
    
    # 显示配置信息
    print(f"目标数字: {config.target_digit}")
    print(f"目标比例: {config.target_ratio:.0%}")
    print(f"序列长度: {config.sequence_length}")
    print(f"间隔范围: {config.isi_range[0]}-{config.isi_range[1]}ms")
    print(f"文件已保存: {filename}")


def example_multiple_sequences():
    """批量生成多个序列"""
    print("\n" + "=" * 50)
    print("示例3: 批量生成多个序列")
    print("=" * 50)
    
    # 不同目标数字的配置
    target_digits = [3, 5, 7]
    
    for i, target in enumerate(target_digits, 1):
        config = CPTConfig(
            target_digit=target,
            target_ratio=0.3,
            sequence_length=40
        )
        
        generator = CPTSequenceGenerator(config)
        sequence = generator.generate_sequence()
        
        filename = f'sequence_target_{target}.json'
        generator.save_sequence(sequence, filename)
        
        print(f"序列{i} (目标数字{target}): {filename}")


def example_load_and_analyze():
    """加载和分析序列示例"""
    print("\n" + "=" * 50)
    print("示例4: 加载和分析序列")
    print("=" * 50)
    
    try:
        # 加载之前生成的序列
        sequence = CPTSequenceGenerator.load_sequence('basic_sequence.json')
        
        # 分析序列内容
        digits = sequence['digits']
        is_target = sequence['is_target']
        intervals = sequence['intervals']
        
        print("序列前10个数字:")
        for i in range(min(10, len(digits))):
            target_str = "目标" if is_target[i] else "非目标"
            print(f"  {i+1}: 数字{digits[i]} ({target_str}), 间隔{intervals[i]}ms")
        
        # 统计分析
        target_positions = [i for i, t in enumerate(is_target) if t]
        print(f"\n目标数字出现位置: {target_positions[:5]}..." if len(target_positions) > 5 else f"\n目标数字出现位置: {target_positions}")
        
    except FileNotFoundError:
        print("请先运行基础使用示例生成序列文件")


def example_error_handling():
    """错误处理示例"""
    print("\n" + "=" * 50)
    print("示例5: 错误处理")
    print("=" * 50)
    
    try:
        # 尝试使用无效配置
        invalid_config = CPTConfig(
            target_digit=10,  # 无效数字
            target_ratio=0.3
        )
        generator = CPTSequenceGenerator(invalid_config)
        
    except ValueError as e:
        print(f"捕获到配置错误: {e}")
    
    try:
        # 尝试使用无效比例
        invalid_config2 = CPTConfig(
            target_digit=5,
            target_ratio=1.5  # 无效比例
        )
        generator = CPTSequenceGenerator(invalid_config2)
        
    except ValueError as e:
        print(f"捕获到比例错误: {e}")


def example_performance_test():
    """性能测试示例"""
    print("\n" + "=" * 50)
    print("示例6: 性能测试")
    print("=" * 50)
    
    import time
    
    # 测试不同序列长度的生成时间
    lengths = [40, 80, 160]
    
    for length in lengths:
        config = CPTConfig(sequence_length=length)
        generator = CPTSequenceGenerator(config)
        
        start_time = time.time()
        sequence = generator.generate_sequence(max_attempts=50)
        end_time = time.time()
        
        generation_time = end_time - start_time
        print(f"序列长度{length}: 生成耗时 {generation_time:.3f}秒")


def main():
    """运行所有示例"""
    print("CPT序列生成器使用示例")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_custom_config()
        example_multiple_sequences()
        example_load_and_analyze()
        example_error_handling()
        example_performance_test()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        print("生成的文件:")
        print("- basic_sequence.json")
        print("- custom_sequence.json")
        print("- sequence_target_3.json")
        print("- sequence_target_5.json")
        print("- sequence_target_7.json")
        print("\n使用 verify_sequence.py 验证生成的序列:")
        print("python verify_sequence.py basic_sequence.json")
        print("python verify_sequence.py basic_sequence.json --simulate")
        
    except Exception as e:
        print(f"运行示例时发生错误: {e}")


if __name__ == "__main__":
    main()