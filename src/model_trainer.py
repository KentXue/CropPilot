#!/usr/bin/env python3
"""
模型训练框架
实现ModelTrainer类，支持训练进度监控、早停和学习率调度
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

from src.model_architecture import PlantDiseaseEfficientNet, ModelFactory

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # 模型配置
    model_type: str = 'efficientnet'
    model_name: str = 'efficientnet-b4'
    num_classes: int = 38
    pretrained: bool = True
    
    # 优化器配置
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    momentum: float = 0.9
    
    # 学习率调度
    scheduler_type: str = 'cosine'  # 'step', 'cosine', 'plateau'
    step_size: int = 30
    gamma: float = 0.1
    min_lr: float = 1e-6
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # 保存配置
    save_dir: str = 'checkpoints'
    save_best_only: bool = True
    save_frequency: int = 5
    
    # 其他配置
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # 数据增强
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    learning_rate: float = 0.0
    epoch_time: float = 0.0
    best_val_acc: float = 0.0
    best_epoch: int = 0

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        初始化早停
        
        Args:
            patience: 耐心值（多少个epoch没有改善就停止）
            min_delta: 最小改善阈值
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前指标值
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """判断当前分数是否更好"""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        初始化标签平滑损失
        
        Args:
            num_classes: 类别数
            smoothing: 平滑参数
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class MixupCutmixLoss:
    """Mixup和Cutmix损失函数"""
    
    def __init__(self, criterion: nn.Module):
        """
        初始化Mixup/Cutmix损失
        
        Args:
            criterion: 基础损失函数
        """
        self.criterion = criterion
    
    def __call__(self, pred: torch.Tensor, target_a: torch.Tensor, 
                 target_b: torch.Tensor, lam: float) -> torch.Tensor:
        """计算混合损失"""
        return lam * self.criterion(pred, target_a) + (1 - lam) * self.criterion(pred, target_b)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("需要安装必要依赖")
        
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.training_history = []
        
        # 早停
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                mode='max'
            )
        else:
            self.early_stopping = None
        
        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.save_dir / 'logs')
        
        logger.info(f"ModelTrainer初始化完成 - 设备: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """设置设备"""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logger.info("使用CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def setup_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        """
        设置模型
        
        Args:
            model: 外部提供的模型，如果为None则根据配置创建
            
        Returns:
            设置好的模型
        """
        if model is None:
            # 根据配置创建模型
            model = ModelFactory.create_model(
                model_type=self.config.model_type,
                num_classes=self.config.num_classes,
                model_name=self.config.model_name,
                pretrained=self.config.pretrained
            )
        
        self.model = model.to(self.device)
        
        # 设置优化器
        self._setup_optimizer()
        
        # 设置学习率调度器
        self._setup_scheduler()
        
        # 设置损失函数
        self._setup_criterion()
        
        # 设置混合精度
        if self.config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"模型设置完成 - 参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def _setup_optimizer(self):
        """设置优化器"""
        if self.config.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        if self.config.scheduler_type.lower() == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type.lower() == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.gamma,
                patience=self.config.patience // 2,
                min_lr=self.config.min_lr
            )
        else:
            raise ValueError(f"不支持的调度器类型: {self.config.scheduler_type}")
    
    def _setup_criterion(self):
        """设置损失函数"""
        if self.config.label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                num_classes=self.config.num_classes,
                smoothing=self.config.label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.criterion = self.criterion.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史
        """
        logger.info("开始训练...")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # 学习率调度
            if self.config.scheduler_type.lower() == 'plateau':
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start_time
            
            # 记录指标
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=current_lr,
                epoch_time=epoch_time,
                best_val_acc=self.best_val_acc,
                best_epoch=self.best_epoch
            )
            
            self.training_history.append(asdict(metrics))
            
            # 更新最佳结果
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                metrics.best_val_acc = self.best_val_acc
                metrics.best_epoch = self.best_epoch
                
                # 保存最佳模型
                if self.config.save_best_only:
                    self.save_checkpoint('best_model.pth', metrics)
            
            # 定期保存
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', metrics)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 打印进度
            logger.info(
                f'Epoch {epoch+1}/{self.config.num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                f'LR: {current_lr:.6f} - Time: {epoch_time:.2f}s'
            )
            
            # 早停检查
            if self.early_stopping is not None:
                if self.early_stopping(val_acc):
                    logger.info(f'早停触发，在第 {epoch+1} 轮停止训练')
                    break
        
        total_time = time.time() - start_time
        logger.info(f'训练完成 - 总时间: {total_time:.2f}s, 最佳验证准确率: {self.best_val_acc:.2f}%')
        
        # 保存训练历史
        self.save_training_history()
        
        # 关闭TensorBoard
        self.writer.close()
        
        return {
            'training_history': self.training_history,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_time': total_time
        }
    
    def save_checkpoint(self, filename: str, metrics: TrainingMetrics):
        """保存检查点"""
        checkpoint = {
            'epoch': metrics.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'config': asdict(self.config),
            'metrics': asdict(metrics)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.save_dir / filename)
        logger.info(f'检查点已保存: {filename}')
    
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint_path = self.save_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_epoch = checkpoint['best_epoch']
        
        logger.info(f'检查点已加载: {filename}')
        
        return checkpoint
    
    def save_training_history(self):
        """保存训练历史"""
        history_file = self.save_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.training_history:
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]
        train_accs = [h['train_acc'] for h in self.training_history]
        val_accs = [h['val_acc'] for h in self.training_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        ax1.plot(epochs, train_losses, label='Train Loss')
        ax1.plot(epochs, val_losses, label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, train_accs, label='Train Acc')
        ax2.plot(epochs, val_accs, label='Val Acc')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        lrs = [h['learning_rate'] for h in self.training_history]
        ax3.plot(epochs, lrs)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 每轮时间
        times = [h['epoch_time'] for h in self.training_history]
        ax4.plot(epochs, times)
        ax4.set_title('Epoch Time')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

# 便捷函数
def create_trainer(config: TrainingConfig) -> ModelTrainer:
    """创建模型训练器"""
    return ModelTrainer(config)

def create_default_config(**kwargs) -> TrainingConfig:
    """创建默认训练配置"""
    config = TrainingConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

if __name__ == "__main__":
    # 测试模型训练框架
    print("🧪 模型训练框架测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试训练配置
        print("📋 测试训练配置...")
        config = create_default_config(
            num_epochs=5,
            batch_size=16,
            learning_rate=0.001,
            model_name='efficientnet-b4',
            pretrained=False  # 避免下载
        )
        
        print(f"✅ 训练配置创建成功:")
        print(f"   轮数: {config.num_epochs}")
        print(f"   批大小: {config.batch_size}")
        print(f"   学习率: {config.learning_rate}")
        print(f"   模型: {config.model_name}")
        
        # 测试训练器创建
        print(f"\n🔧 测试训练器创建...")
        trainer = create_trainer(config)
        
        print(f"✅ 训练器创建成功:")
        print(f"   设备: {trainer.device}")
        print(f"   保存目录: {trainer.save_dir}")
        
        # 测试模型设置
        print(f"\n🏗️ 测试模型设置...")
        model = trainer.setup_model()
        
        print(f"✅ 模型设置完成:")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   优化器: {type(trainer.optimizer).__name__}")
        print(f"   调度器: {type(trainer.scheduler).__name__}")
        print(f"   损失函数: {type(trainer.criterion).__name__}")
        
        # 测试早停机制
        print(f"\n⏹️ 测试早停机制...")
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # 模拟验证准确率变化
        val_accs = [0.85, 0.87, 0.86, 0.86, 0.85, 0.84]
        for i, acc in enumerate(val_accs):
            should_stop = early_stopping(acc)
            print(f"   Epoch {i+1}: Val Acc = {acc:.2f}, Should Stop = {should_stop}")
            if should_stop:
                break
        
        # 测试标签平滑损失
        print(f"\n🎯 测试标签平滑损失...")
        label_smooth_loss = LabelSmoothingLoss(num_classes=38, smoothing=0.1)
        
        # 创建测试数据
        pred = torch.randn(4, 38)
        target = torch.randint(0, 38, (4,))
        
        loss = label_smooth_loss(pred, target)
        print(f"✅ 标签平滑损失测试完成:")
        print(f"   输入形状: {pred.shape}")
        print(f"   目标形状: {target.shape}")
        print(f"   损失值: {loss.item():.4f}")
        
        # 清理
        trainer.writer.close()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 模型训练框架测试完成")