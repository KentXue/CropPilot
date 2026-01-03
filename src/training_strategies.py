#!/usr/bin/env python3
"""
训练策略优化模块
实现类别权重平衡、渐进式训练、超参数优化等高级训练策略
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import Counter
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProgressiveTrainingConfig:
    """渐进式训练配置"""
    stages: List[Dict[str, Any]]
    warmup_epochs: int = 5
    freeze_backbone_epochs: int = 10
    
class ClassBalanceStrategy:
    """类别平衡策略"""
    
    def __init__(self, strategy: str = 'weighted_loss'):
        """
        初始化类别平衡策略
        
        Args:
            strategy: 平衡策略 ('weighted_loss', 'weighted_sampling', 'focal_loss')
        """
        self.strategy = strategy
        self.class_weights = None
        self.sample_weights = None
        
    def compute_class_weights(self, labels: List[int], num_classes: int) -> torch.Tensor:
        """计算类别权重"""
        # 统计每个类别的样本数
        class_counts = Counter(labels)
        
        # 确保所有类别都有计数
        for i in range(num_classes):
            if i not in class_counts:
                class_counts[i] = 1
        
        # 计算权重
        total_samples = len(labels)
        weights = []
        
        for i in range(num_classes):
            count = class_counts[i]
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        self.class_weights = torch.FloatTensor(weights)
        logger.info(f"类别权重计算完成，权重范围: [{self.class_weights.min():.3f}, {self.class_weights.max():.3f}]")
        
        return self.class_weights
    
    def compute_sample_weights(self, labels: List[int]) -> torch.Tensor:
        """计算样本权重用于加权采样"""
        if self.class_weights is None:
            raise ValueError("请先调用compute_class_weights")
        
        sample_weights = []
        for label in labels:
            sample_weights.append(self.class_weights[label].item())
        
        self.sample_weights = torch.FloatTensor(sample_weights)
        return self.sample_weights
    
    def create_weighted_sampler(self, labels: List[int], num_classes: int) -> WeightedRandomSampler:
        """创建加权采样器"""
        self.compute_class_weights(labels, num_classes)
        sample_weights = self.compute_sample_weights(labels)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        logger.info("加权采样器创建完成")
        return sampler

class FocalLoss(nn.Module):
    """Focal Loss实现"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 减少方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ProgressiveTrainer:
    """渐进式训练器"""
    
    def __init__(self, model: nn.Module, config: ProgressiveTrainingConfig):
        """
        初始化渐进式训练器
        
        Args:
            model: 要训练的模型
            config: 渐进式训练配置
        """
        self.model = model
        self.config = config
        self.current_stage = 0
        
    def freeze_backbone(self):
        """冻结骨干网络"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            logger.info("骨干网络已冻结")
        
    def unfreeze_backbone(self):
        """解冻骨干网络"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            logger.info("骨干网络已解冻")
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """获取阶段配置"""
        if stage < len(self.config.stages):
            return self.config.stages[stage]
        else:
            return self.config.stages[-1]  # 使用最后一个阶段的配置
    
    def should_advance_stage(self, epoch: int, val_acc: float) -> bool:
        """判断是否应该进入下一阶段"""
        stage_config = self.get_stage_config(self.current_stage)
        
        # 检查轮数条件
        if epoch >= stage_config.get('min_epochs', 10):
            # 检查准确率条件
            target_acc = stage_config.get('target_accuracy', 0.0)
            if val_acc >= target_acc:
                return True
        
        return False
    
    def advance_stage(self, optimizer: optim.Optimizer) -> bool:
        """进入下一阶段"""
        if self.current_stage < len(self.config.stages) - 1:
            self.current_stage += 1
            stage_config = self.get_stage_config(self.current_stage)
            
            # 更新学习率
            new_lr = stage_config.get('learning_rate', 0.001)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # 解冻策略
            if stage_config.get('unfreeze_backbone', False):
                self.unfreeze_backbone()
            
            logger.info(f"进入训练阶段 {self.current_stage + 1}, 学习率: {new_lr}")
            return True
        
        return False

class GradientAccumulator:
    """梯度累积器"""
    
    def __init__(self, accumulation_steps: int = 4):
        """
        初始化梯度累积器
        
        Args:
            accumulation_steps: 累积步数
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_step(self) -> bool:
        """判断是否应该执行优化步骤"""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        return loss / self.accumulation_steps

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, optimizer: optim.Optimizer, strategy: str = 'cosine_warmup'):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            strategy: 调度策略
        """
        self.optimizer = optimizer
        self.strategy = strategy
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def warmup_lr(self, epoch: int, warmup_epochs: int) -> float:
        """预热学习率"""
        if epoch < warmup_epochs:
            return self.base_lr * (epoch + 1) / warmup_epochs
        return self.base_lr
    
    def cosine_annealing_lr(self, epoch: int, total_epochs: int, min_lr: float = 1e-6) -> float:
        """余弦退火学习率"""
        return min_lr + (self.base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    
    def step_lr(self, epoch: int, step_size: int = 30, gamma: float = 0.1) -> float:
        """阶梯学习率"""
        return self.base_lr * (gamma ** (epoch // step_size))
    
    def update_lr(self, epoch: int, **kwargs):
        """更新学习率"""
        if self.strategy == 'cosine_warmup':
            warmup_epochs = kwargs.get('warmup_epochs', 5)
            total_epochs = kwargs.get('total_epochs', 100)
            
            if epoch < warmup_epochs:
                new_lr = self.warmup_lr(epoch, warmup_epochs)
            else:
                new_lr = self.cosine_annealing_lr(epoch - warmup_epochs, total_epochs - warmup_epochs)
        
        elif self.strategy == 'step':
            new_lr = self.step_lr(epoch, kwargs.get('step_size', 30), kwargs.get('gamma', 0.1))
        
        else:
            new_lr = self.base_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

class TrainingOptimizer:
    """训练优化器"""
    
    def __init__(self):
        """初始化训练优化器"""
        self.strategies = {}
        
    def optimize_batch_size(self, model: nn.Module, device: torch.device, 
                          input_size: Tuple[int, int, int] = (3, 224, 224),
                          max_memory_gb: float = 6.0) -> int:
        """优化批大小"""
        logger.info("开始批大小优化...")
        
        model.eval()
        optimal_batch_size = 1
        
        # 测试不同的批大小
        test_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        
        for batch_size in test_batch_sizes:
            try:
                # 创建测试数据
                test_input = torch.randn(batch_size, *input_size, device=device)
                
                # 前向传播测试
                with torch.no_grad():
                    _ = model(test_input)
                
                # 检查GPU内存使用
                if device.type == 'cuda':
                    memory_used = torch.cuda.memory_allocated(device) / 1024**3
                    if memory_used > max_memory_gb * 0.8:  # 使用80%的显存作为上限
                        break
                
                optimal_batch_size = batch_size
                
                # 清理
                del test_input
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        logger.info(f"优化批大小: {optimal_batch_size}")
        return optimal_batch_size
    
    def create_progressive_config(self, num_classes: int) -> ProgressiveTrainingConfig:
        """创建渐进式训练配置"""
        stages = [
            {
                'name': 'warmup',
                'min_epochs': 5,
                'learning_rate': 0.0001,
                'target_accuracy': 0.1,
                'freeze_backbone': True
            },
            {
                'name': 'fine_tune_head',
                'min_epochs': 10,
                'learning_rate': 0.001,
                'target_accuracy': 0.3,
                'freeze_backbone': True
            },
            {
                'name': 'full_training',
                'min_epochs': 20,
                'learning_rate': 0.0005,
                'target_accuracy': 0.8,
                'unfreeze_backbone': True
            }
        ]
        
        return ProgressiveTrainingConfig(
            stages=stages,
            warmup_epochs=5,
            freeze_backbone_epochs=15
        )
    
    def suggest_hyperparameters(self, dataset_size: int, num_classes: int, 
                              gpu_memory_gb: float) -> Dict[str, Any]:
        """建议超参数"""
        suggestions = {}
        
        # 批大小建议
        if gpu_memory_gb >= 24:
            suggestions['batch_size'] = min(64, dataset_size // 100)
        elif gpu_memory_gb >= 12:
            suggestions['batch_size'] = min(32, dataset_size // 200)
        elif gpu_memory_gb >= 8:
            suggestions['batch_size'] = min(16, dataset_size // 400)
        else:
            suggestions['batch_size'] = min(8, dataset_size // 800)
        
        # 学习率建议
        if dataset_size < 1000:
            suggestions['learning_rate'] = 0.0001
        elif dataset_size < 10000:
            suggestions['learning_rate'] = 0.001
        else:
            suggestions['learning_rate'] = 0.01
        
        # 轮数建议
        if dataset_size < 1000:
            suggestions['num_epochs'] = 100
        elif dataset_size < 10000:
            suggestions['num_epochs'] = 50
        else:
            suggestions['num_epochs'] = 30
        
        # 其他建议
        suggestions['weight_decay'] = 1e-4
        suggestions['dropout_rate'] = 0.3 if num_classes > 20 else 0.2
        suggestions['label_smoothing'] = 0.1 if num_classes > 10 else 0.0
        
        return suggestions

def create_balanced_dataloader(dataset, labels: List[int], batch_size: int, 
                             num_classes: int, strategy: str = 'weighted_sampling') -> DataLoader:
    """创建平衡的数据加载器"""
    balance_strategy = ClassBalanceStrategy(strategy)
    
    if strategy == 'weighted_sampling':
        sampler = balance_strategy.create_weighted_sampler(labels, num_classes)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_progressive_trainer(model: nn.Module, num_classes: int) -> ProgressiveTrainer:
    """创建渐进式训练器"""
    optimizer = TrainingOptimizer()
    config = optimizer.create_progressive_config(num_classes)
    return ProgressiveTrainer(model, config)

if __name__ == "__main__":
    # 测试训练策略优化
    print("🧪 训练策略优化测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试类别平衡策略
        print("📊 测试类别平衡策略...")
        labels = [0] * 100 + [1] * 50 + [2] * 200  # 不平衡数据
        balance_strategy = ClassBalanceStrategy()
        
        class_weights = balance_strategy.compute_class_weights(labels, 3)
        print(f"✅ 类别权重: {class_weights}")
        
        # 测试Focal Loss
        print(f"\n🎯 测试Focal Loss...")
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        
        # 创建测试数据
        pred = torch.randn(10, 3)
        target = torch.randint(0, 3, (10,))
        
        loss = focal_loss(pred, target)
        print(f"✅ Focal Loss: {loss.item():.4f}")
        
        # 测试梯度累积
        print(f"\n📈 测试梯度累积...")
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        for i in range(10):
            should_step = accumulator.should_step()
            if should_step:
                print(f"   步骤 {i+1}: 执行优化")
        
        # 测试训练优化器
        print(f"\n⚙️ 测试训练优化器...")
        optimizer = TrainingOptimizer()
        
        suggestions = optimizer.suggest_hyperparameters(
            dataset_size=10000,
            num_classes=38,
            gpu_memory_gb=6.0
        )
        
        print(f"✅ 超参数建议:")
        for key, value in suggestions.items():
            print(f"   {key}: {value}")
        
        # 测试渐进式训练配置
        print(f"\n🚀 测试渐进式训练配置...")
        config = optimizer.create_progressive_config(38)
        
        print(f"✅ 渐进式训练阶段:")
        for i, stage in enumerate(config.stages):
            print(f"   阶段 {i+1}: {stage['name']} - LR: {stage['learning_rate']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 训练策略优化测试完成")