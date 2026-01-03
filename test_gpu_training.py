#!/usr/bin/env python3
"""
GPU训练测试脚本
测试GPU加速的训练流程
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

from src.model_architecture import create_plant_disease_model, ModelFactory
from src.model_trainer import ModelTrainer, create_default_config
from src.model_evaluator import create_evaluator

def test_gpu_training():
    """测试GPU训练"""
    print("🚀 GPU训练测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ GPU不可用，无法进行GPU训练测试")
        return False
    
    try:
        # 创建虚拟数据集
        print("创建测试数据集...")
        num_samples = 160  # 20个批次，每批8个样本
        images = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 38, (num_samples,))
        
        train_size = int(0.7 * num_samples)
        val_size = num_samples - train_size
        
        train_dataset = TensorDataset(images[:train_size], labels[:train_size])
        val_dataset = TensorDataset(images[train_size:], labels[train_size:])
        
        # 创建数据加载器（使用GPU优化设置）
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=True,
            num_workers=0,  # Windows上设为0避免多进程问题
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=8, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # 创建GPU训练配置
        config = create_default_config(
            num_epochs=3,
            batch_size=8,
            learning_rate=0.001,
            model_name='efficientnet-b4',
            pretrained=False,  # 避免下载
            save_dir='test_gpu_checkpoints',
            device='cuda',
            mixed_precision=True,
            early_stopping=False
        )
        
        print(f"配置信息:")
        print(f"  设备: {config.device}")
        print(f"  批大小: {config.batch_size}")
        print(f"  混合精度: {config.mixed_precision}")
        
        # 创建训练器
        trainer = ModelTrainer(config)
        model = trainer.setup_model()
        
        print(f"模型信息:")
        print(f"  设备: {next(model.parameters()).device}")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试GPU内存使用
        print(f"\n训练前GPU内存:")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已缓存: {cached:.2f} GB")
        
        # 执行训练
        print(f"\n开始GPU训练...")
        start_time = time.time()
        
        training_results = trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        
        # 训练后GPU内存使用
        print(f"\n训练后GPU内存:")
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已缓存: {cached:.2f} GB")
        
        print(f"\n✅ GPU训练完成:")
        print(f"  训练时间: {training_time:.2f}秒")
        print(f"  最佳验证准确率: {training_results['best_val_acc']:.2f}%")
        print(f"  使用设备: {trainer.device}")
        
        # 清理
        del model, trainer
        torch.cuda.empty_cache()
        
        # 清理测试文件
        import shutil
        if os.path.exists('test_gpu_checkpoints'):
            shutil.rmtree('test_gpu_checkpoints')
        
        return True
        
    except Exception as e:
        print(f"❌ GPU训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        return False

def compare_cpu_gpu_performance():
    """比较CPU和GPU性能"""
    print("\n⚡ CPU vs GPU性能对比")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ GPU不可用，无法进行性能对比")
        return
    
    try:
        # 创建测试模型和数据
        model_cpu = create_plant_disease_model('efficientnet', pretrained=False)
        model_gpu = create_plant_disease_model('efficientnet', pretrained=False).cuda()
        
        test_data_cpu = torch.randn(8, 3, 224, 224)
        test_data_gpu = test_data_cpu.cuda()
        
        # CPU性能测试
        model_cpu.eval()
        with torch.no_grad():
            # 预热
            for _ in range(5):
                _ = model_cpu(test_data_cpu)
            
            # 测试
            start_time = time.time()
            for _ in range(20):
                _ = model_cpu(test_data_cpu)
            cpu_time = (time.time() - start_time) / 20
        
        # GPU性能测试
        model_gpu.eval()
        with torch.no_grad():
            # 预热
            for _ in range(5):
                _ = model_gpu(test_data_gpu)
            torch.cuda.synchronize()
            
            # 测试
            start_time = time.time()
            for _ in range(20):
                _ = model_gpu(test_data_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start_time) / 20
        
        speedup = cpu_time / gpu_time
        
        print(f"性能对比结果 (批大小=8):")
        print(f"  CPU时间: {cpu_time*1000:.1f}ms/batch")
        print(f"  GPU时间: {gpu_time*1000:.1f}ms/batch")
        print(f"  加速比: {speedup:.1f}x")
        
        if speedup > 1:
            print(f"🚀 GPU比CPU快 {speedup:.1f} 倍!")
        else:
            print(f"⚠️  GPU性能未达到预期")
        
        # 清理
        del model_cpu, model_gpu, test_data_cpu, test_data_gpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 性能对比失败: {e}")
        torch.cuda.empty_cache()

def main():
    """主函数"""
    print("🧪 GPU训练测试套件")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        return
    
    if not torch.cuda.is_available():
        print("❌ GPU不可用，请先运行 check_gpu.py 检查GPU设置")
        return
    
    print(f"GPU信息: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试GPU训练
    success = test_gpu_training()
    
    if success:
        # 性能对比
        compare_cpu_gpu_performance()
        
        print("\n🎉 GPU训练测试全部通过!")
        print("现在可以使用以下命令进行GPU训练:")
        print("python train_model.py --config gpu_training_config.json")
    else:
        print("\n❌ GPU训练测试失败")

if __name__ == "__main__":
    main()