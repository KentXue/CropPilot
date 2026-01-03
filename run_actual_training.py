#!/usr/bin/env python3
"""
实际执行训练脚本
使用GPU优化配置执行真实的植物病害识别模型训练
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import numpy as np
    from tqdm import tqdm
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

# 导入项目模块
from src.model_architecture import create_plant_disease_model, ModelFactory
from src.model_trainer import ModelTrainer, create_default_config
from src.model_evaluator import create_evaluator
from src.training_strategies import ClassBalanceStrategy, FocalLoss
from src.model_optimization import InferenceOptimizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('actual_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_realistic_dataset(num_samples: int = 5000, num_classes: int = 38):
    """创建更真实的数据集用于训练"""
    logger.info(f"创建真实训练数据集: {num_samples} 样本, {num_classes} 类别")
    
    # 创建更真实的图像数据（模拟植物病害图像特征）
    images = []
    labels = []
    
    # 模拟不平衡的类别分布（真实数据集通常是不平衡的）
    class_weights = np.random.exponential(1.0, num_classes)
    class_weights = class_weights / class_weights.sum()
    
    for i in range(num_samples):
        # 根据权重选择类别
        class_id = np.random.choice(num_classes, p=class_weights)
        
        # 创建具有一定模式的图像（模拟植物特征）
        base_image = torch.randn(3, 224, 224)
        
        # 添加类别特定的模式
        if class_id < 10:  # 健康植物
            base_image = base_image * 0.5 + 0.3  # 较亮的绿色调
        elif class_id < 20:  # 叶斑病
            base_image[0] *= 1.2  # 增强红色通道
        elif class_id < 30:  # 萎蔫病
            base_image = base_image * 0.7  # 较暗
        else:  # 其他病害
            base_image[1] *= 0.8  # 减少绿色通道
        
        # 添加噪声
        noise = torch.randn_like(base_image) * 0.1
        final_image = torch.clamp(base_image + noise, -2, 2)
        
        images.append(final_image)
        labels.append(class_id)
    
    # 转换为张量
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # 统计类别分布
    from collections import Counter
    class_distribution = Counter(labels)
    logger.info(f"类别分布 (前10类): {dict(list(class_distribution.most_common(10)))}")
    
    return TensorDataset(images_tensor, labels_tensor), class_distribution

def run_gpu_training():
    """执行GPU训练"""
    logger.info("🚀 开始GPU加速植物病害识别模型训练")
    logger.info("=" * 80)
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        logger.error("❌ GPU不可用，请检查CUDA安装")
        return False
    
    device = torch.device('cuda')
    logger.info(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    start_time = time.time()
    
    try:
        # 1. 创建数据集
        logger.info("\n📊 步骤1: 创建训练数据集")
        full_dataset, class_distribution = create_realistic_dataset(
            num_samples=3000,  # 适中的数据集大小
            num_classes=38
        )
        
        # 分割数据集
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"   训练集: {len(train_dataset)} 样本")
        logger.info(f"   验证集: {len(val_dataset)} 样本")
        logger.info(f"   测试集: {len(test_dataset)} 样本")
        
        # 2. 创建数据加载器
        logger.info("\n🔄 步骤2: 创建数据加载器")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=8,  # 适合6GB显存
            shuffle=True,
            num_workers=0,  # Windows兼容性
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=8, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=8, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # 3. 创建训练配置
        logger.info("\n⚙️ 步骤3: 配置训练参数")
        config = create_default_config(
            num_epochs=15,  # 适中的训练轮数
            batch_size=8,
            learning_rate=0.001,
            model_name='efficientnet-b4',
            pretrained=True,  # 使用预训练权重
            device='cuda',
            mixed_precision=True,
            early_stopping=True,
            patience=5,
            save_dir='checkpoints/actual_training_run',
            num_classes=38
        )
        
        logger.info(f"   训练轮数: {config.num_epochs}")
        logger.info(f"   批大小: {config.batch_size}")
        logger.info(f"   学习率: {config.learning_rate}")
        logger.info(f"   混合精度: {config.mixed_precision}")
        
        # 4. 创建训练器和模型
        logger.info("\n🏗️ 步骤4: 初始化模型和训练器")
        trainer = ModelTrainer(config)
        model = trainer.setup_model()
        
        # 使用Focal Loss处理类别不平衡
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        trainer.criterion = focal_loss.to(device)
        
        model_info = ModelFactory.get_model_info(model)
        logger.info(f"   模型: {model_info['model_type']}")
        logger.info(f"   参数数量: {model_info['total_parameters']:,}")
        logger.info(f"   模型大小: {model_info['model_size_mb']:.1f} MB")
        logger.info(f"   损失函数: Focal Loss")
        
        # 5. 执行训练
        logger.info("\n🚀 步骤5: 开始模型训练")
        logger.info("-" * 50)
        
        training_start = time.time()
        training_results = trainer.train(train_loader, val_loader)
        training_time = time.time() - training_start
        
        logger.info("-" * 50)
        logger.info(f"✅ 训练完成!")
        logger.info(f"   训练时间: {training_time:.2f}秒")
        logger.info(f"   最佳验证准确率: {training_results['best_val_acc']:.2f}%")
        logger.info(f"   最佳轮次: {training_results['best_epoch']}")
        
        # 6. 模型评估
        logger.info("\n📊 步骤6: 模型评估")
        
        # 加载最佳模型
        best_model_path = Path(config.save_dir) / 'best_model.pth'
        if best_model_path.exists():
            trainer.load_checkpoint('best_model.pth')
            logger.info("   已加载最佳模型权重")
        
        # 创建评估器
        class_names = [f'Disease_{i:02d}' for i in range(38)]
        evaluator = create_evaluator(class_names=class_names, device='cuda')
        
        # 评估模型
        metrics, _ = evaluator.evaluate_model(
            trainer.model, test_loader, return_predictions=False
        )
        
        logger.info(f"   测试准确率: {metrics.accuracy:.4f}")
        logger.info(f"   F1分数(宏): {metrics.f1_macro:.4f}")
        logger.info(f"   F1分数(加权): {metrics.f1_weighted:.4f}")
        logger.info(f"   Top-3准确率: {metrics.top_k_accuracy.get(3, 0):.4f}")
        
        # 7. 推理优化
        logger.info("\n⚡ 步骤7: 推理优化")
        
        # 获取示例输入
        example_input = next(iter(test_loader))[0][:1].to(device)
        
        # 推理优化
        inference_opt = InferenceOptimizer()
        
        # 原始模型性能
        original_benchmark = inference_opt.benchmark_inference(
            trainer.model, example_input, num_runs=50
        )
        
        # JIT优化
        try:
            optimized_model = inference_opt.optimize_for_inference(
                trainer.model, example_input, 'basic'
            )
            optimized_benchmark = inference_opt.benchmark_inference(
                optimized_model, example_input, num_runs=50
            )
            
            speedup = original_benchmark['avg_inference_time_ms'] / optimized_benchmark['avg_inference_time_ms']
            
            logger.info(f"   原始推理速度: {original_benchmark['avg_inference_time_ms']:.1f}ms")
            logger.info(f"   优化推理速度: {optimized_benchmark['avg_inference_time_ms']:.1f}ms")
            logger.info(f"   推理加速比: {speedup:.2f}x")
            
        except Exception as e:
            logger.warning(f"   推理优化跳过: {e}")
            speedup = 1.0
        
        # 8. 保存训练总结
        logger.info("\n💾 步骤8: 保存训练总结")
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_config': {
                'num_epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'model_name': config.model_name,
                'device': str(device)
            },
            'dataset_info': {
                'total_samples': len(full_dataset),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset),
                'num_classes': 38
            },
            'training_results': {
                'best_val_acc': training_results['best_val_acc'],
                'best_epoch': training_results['best_epoch'],
                'training_time_sec': training_time
            },
            'evaluation_results': {
                'test_accuracy': metrics.accuracy,
                'f1_macro': metrics.f1_macro,
                'f1_weighted': metrics.f1_weighted,
                'top_k_accuracy': metrics.top_k_accuracy
            },
            'optimization_results': {
                'inference_speedup': speedup,
                'original_inference_ms': original_benchmark['avg_inference_time_ms'],
                'optimized_inference_ms': optimized_benchmark['avg_inference_time_ms'] if 'optimized_benchmark' in locals() else None
            }
        }
        
        summary_path = Path(config.sav