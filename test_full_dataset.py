#!/usr/bin/env python3
"""
测试完整数据集加载
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.dataset_manager import get_dataset_manager

def test_full_dataset():
    """测试完整数据集加载"""
    print("🧪 测试完整数据集加载")
    print("=" * 50)
    
    manager = get_dataset_manager()
    
    # 测试color数据集（不限制样本数）
    try:
        print("🔍 加载完整color数据集...")
        dataset = manager.load_dataset('color')  # 不限制样本数
        info = dataset.get_dataset_info()
        
        print(f"✅ 加载成功:")
        print(f"   数据集: {info['name']}")
        print(f"   样本数: {info['total_samples']}")
        print(f"   类别数: {info['num_classes']}")
        
        # 显示类别分布前10个
        distribution = info['class_distribution']
        sorted_classes = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        print(f"   类别分布 (前10个):")
        for class_name, count in sorted_classes[:10]:
            print(f"     {class_name}: {count} 张")
        
        # 检查数据集平衡性
        counts = list(distribution.values())
        if counts:
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"   不平衡比例: {imbalance_ratio:.1f}")
            
            if imbalance_ratio > 10:
                print("   ⚠️  类别严重不平衡")
            elif imbalance_ratio > 5:
                print("   ⚠️  类别中度不平衡")
            else:
                print("   ✅ 类别相对平衡")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_full_dataset()
    if success:
        print(f"\n🎉 完整数据集测试成功!")
        print("📋 下一步: 可以开始任务1.2 - 实现PlantVillage数据集加载器")
    else:
        print(f"\n❌ 测试失败，请检查数据集路径和格式")