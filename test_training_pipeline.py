#!/usr/bin/env python3
"""
训练管道测试脚本
快速测试训练流程的各个组件
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
from src.image_preprocessing import PlantDiseasePreprocessor, PreprocessingMode

def create_dummy_dataset(num_samples: int = 1000, num_classes: int = 38):
    """创建虚拟数据集用于测试"""
    print(f"创建虚拟数据集: {num_samples} 样本, {num_classes} 类别")
    
    # 创建随机图像数据
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 分割数据集
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset = TensorDataset(images[:train_size], labels[:train_size])
    val_dataset = TensorDataset(images[train_size:train_size+val_size], labels[train_size:train_size+val_size])
    test_dataset = TensorDataset(images[train_size+val_size:], labels[train_size+val_size:])
    
    return train_dataset, val_dataset, test_dataset

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    
    try:
        # 创建EfficientNet模型
        model = create_plant_disease_model(
            model_type='efficientnet',
            num_classes=38,
            model_name='efficientnet-b4',
            pretrained=False  # 避免下载预训练权重
        )
        
        model_info = ModelFactory.get_model_info(model)
        
        print(f"✅ 模型创建成功:")
        print(f"   模型类型: {model_info['model_type']}")
        print(f"   参数数量: {model_info['total_parameters']:,}")
        print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"   前向传播测试: 输入 {test_input.shape} -> 输出 {output.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None

def test_trainer_setup():
    """测试训练器设置"""
    print("\n🔧 测试训练器设置...")
    
    try:
        # 创建训练配置
        config = create_default_config(
            num_epochs=3,  # 短训练用于测试
            batch_size=16,
            learning_rate=0.001,
            model_name='efficientnet-b4',
            pretrained=False,
            save_dir='test_checkpoints',
            mixed_precision=False  # 避免可能的兼容性问题
        )
        
        # 创建训练器
        trainer = ModelTrainer(config)
        
        # 设置模型
        model = trainer.setup_model()
        
        print(f"✅ 训练器设置成功:")
        print(f"   设备: {trainer.device}")
        print(f"   优化器: {type(trainer.optimizer).__name__}")
        print(f"   调度器: {type(trainer.scheduler).__name__}")
        print(f"   损失函数: {type(trainer.criterion).__name__}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ 训练器设置失败: {e}")
        return None

def test_evaluator_setup():
    """测试评估器设置"""
    print("\n📊 测试评估器设置...")
    
    try:
        # 创建类别名称
        class_names = [f'Disease_{i:02d}' for i in range(38)]
        
        # 创建评估器
        evaluator = create_evaluator(class_names=class_names)
        
        print(f"✅ 评估器设置成功:")
        print(f"   设备: {evaluator.device}")
        print(f"   类别数: {len(class_names)}")
        
        return evaluator, class_names
        
    except Exception as e:
        print(f"❌ 评估器设置失败: {e}")
        return None, None

def test_preprocessing():
    """测试图像预处理"""
    print("\n🖼️ 测试图像预处理...")
    
    try:
        # 创建预处理器
        train_preprocessor = PlantDiseasePreprocessor(
            input_size=(224, 224),
            mode=PreprocessingMode.TRAINING
        )
        
        val_preprocessor = PlantDiseasePreprocessor(
            input_size=(224, 224),
            mode=PreprocessingMode.VALIDATION
        )
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 测试预处理
        train_processed = train_preprocessor.preprocess_image(test_image)
        val_processed = val_preprocessor.preprocess_image(test_image)
        
        print(f"✅ 图像预处理成功:")
        print(f"   训练模式输出: {train_processed.shape}")
        print(f"   验证模式输出: {val_processed.shape}")
        print(f"   数值范围: [{train_processed.min():.3f}, {train_processed.max():.3f}]")
        
        return train_preprocessor, val_preprocessor
        
    except Exception as e:
        print(f"❌ 图像预处理失败: {e}")
        return None, None

def test_mini_training():
    """测试迷你训练流程"""
    print("\n🚀 测试迷你训练流程...")
    
    try:
        # 创建虚拟数据集
        train_dataset, val_dataset, test_dataset = create_dummy_dataset(
            num_samples=200, num_classes=38
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 创建训练配置
        config = create_default_config(
            num_epochs=2,  # 只训练2轮
            batch_size=16,
            learning_rate=0.01,
            model_name='efficientnet-b4',
            pretrained=False,
            save_dir='test_checkpoints',
            early_stopping=False,  # 关闭早停
            mixed_precision=False
        )
        
        # 创建训练器
        trainer = ModelTrainer(config)
        model = trainer.setup_model()
        
        print(f"开始迷你训练...")
        start_time = time.time()
        
        # 执行训练
        training_results = trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        
        print(f"✅ 迷你训练完成:")
        print(f"   训练时间: {training_time:.2f}秒")
        print(f"   最佳验证准确率: {training_results['best_val_acc']:.2f}%")
        print(f"   最佳轮次: {training_results['best_epoch']}")
        
        # 测试评估
        class_names = [f'Disease_{i:02d}' for i in range(38)]
        evaluator = create_evaluator(class_names=class_names)
        
        metrics, _ = evaluator.evaluate_model(
            model, test_loader, return_predictions=False
        )
        
        print(f"✅ 模型评估完成:")
        print(f"   测试准确率: {metrics.accuracy:.4f}")
        print(f"   F1分数: {metrics.f1_macro:.4f}")
        
        # 清理测试文件
        import shutil
        if os.path.exists('test_checkpoints'):
            shutil.rmtree('test_checkpoints')
        
        return True
        
    except Exception as e:
        print(f"❌ 迷你训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 植物病害识别训练管道测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        return
    
    success_count = 0
    total_tests = 5
    
    # 测试1: 模型创建
    if test_model_creation() is not None:
        success_count += 1
    
    # 测试2: 训练器设置
    if test_trainer_setup() is not None:
        success_count += 1
    
    # 测试3: 评估器设置
    evaluator, class_names = test_evaluator_setup()
    if evaluator is not None:
        success_count += 1
    
    # 测试4: 图像预处理
    train_prep, val_prep = test_preprocessing()
    if train_prep is not None:
        success_count += 1
    
    # 测试5: 迷你训练
    if test_mini_training():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"测试完成: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！训练管道准备就绪")
    else:
        print(f"⚠️  {total_tests - success_count} 个测试失败，请检查相关组件")
    
    print("=" * 60)

if __name__ == "__main__":
    main()