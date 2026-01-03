#!/usr/bin/env python3
"""
优化版植物病害识别模型训练脚本
整合所有训练策略优化和性能调优功能
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    from tqdm import tqdm
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

# 导入项目模块
from src.model_architecture import create_plant_disease_model, ModelFactory
from src.model_trainer import ModelTrainer, TrainingConfig, create_default_config
from src.model_evaluator import ModelEvaluator, create_evaluator
from src.image_preprocessing import PlantDiseasePreprocessor, PreprocessingMode
from src.training_strategies import (
    ClassBalanceStrategy, FocalLoss, ProgressiveTrainer, 
    GradientAccumulator, create_balanced_dataloader
)
from src.model_optimization import (
    HyperparameterOptimizer, ModelEnsemble, InferenceOptimizer,
    PerformanceTuner, create_hyperparameter_optimizer
)
from src.dataset_manager import DatasetManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedTrainingManager:
    """优化训练管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化优化训练管理器
        
        Args:
            config_path: 配置文件路径
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("需要安装必要依赖")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.dataset_manager = DatasetManager()
        self.trainer = None
        self.evaluator = None
        self.performance_tuner = PerformanceTuner()
        
        # 数据集信息
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = []
        self.class_labels = []
        
        logger.info("优化训练管理器初始化完成")
    
    def _load_config(self, config_path: Optional[str]) -> TrainingConfig:
        """加载训练配置"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # 创建配置对象
            config = TrainingConfig()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"配置已从文件加载: {config_path}")
        else:
            # 使用GPU优化配置
            config = create_default_config(
                num_epochs=30,
                batch_size=8,  # 适合6GB显存
                learning_rate=0.001,
                model_name='efficientnet-b4',
                pretrained=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                mixed_precision=True,
                early_stopping=True,
                patience=10,
                save_dir='checkpoints/optimized_plant_disease_model'
            )
            logger.info("使用默认GPU优化配置")
        
        return config
    
    def prepare_datasets_with_balance(self) -> Dict[str, Any]:
        """准备带类别平衡的数据集"""
        logger.info("开始准备平衡数据集...")
        
        # 使用虚拟数据集进行训练演示
        # 在实际应用中，这里会加载真实的PlantVillage数据集
        
        # 创建虚拟的植物病害数据
        num_samples = 2000  # 总样本数
        num_classes = 38    # 植物病害类别数
        
        # 模拟类别不平衡的数据分布
        np.random.seed(42)
        class_counts = np.random.randint(20, 100, num_classes)  # 每个类别20-100个样本
        
        # 生成虚拟图像数据和标签
        all_images = []
        all_labels = []
        
        for class_id in range(num_classes):
            count = class_counts[class_id]
            # 生成该类别的虚拟图像（随机噪声）
            class_images = torch.randn(count, 3, 224, 224)
            class_labels = [class_id] * count
            
            all_images.append(class_images)
            all_labels.extend(class_labels)
        
        # 合并所有数据
        all_images = torch.cat(all_images, dim=0)
        
        # 创建类别名称
        self.class_names = [f'植物病害_{i:02d}' for i in range(num_classes)]
        self.class_labels = all_labels
        
        # 数据集分割（分层采样）
        from sklearn.model_selection import train_test_split
        
        # 转换为numpy数组以便分割
        images_np = all_images.numpy()
        labels_np = np.array(all_labels)
        
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            images_np, labels_np, 
            test_size=0.3,
            stratify=labels_np,
            random_state=42
        )
        
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels,
            test_size=0.5,
            stratify=temp_labels,
            random_state=42
        )
        
        # 转换回tensor
        train_images = torch.from_numpy(train_images)
        val_images = torch.from_numpy(val_images)
        test_images = torch.from_numpy(test_images)
        
        train_labels = torch.from_numpy(train_labels)
        val_labels = torch.from_numpy(val_labels)
        test_labels = torch.from_numpy(test_labels)
        
        # 创建数据集对象
        from torch.utils.data import TensorDataset
        
        self.train_dataset = TensorDataset(train_images, train_labels)
        self.val_dataset = TensorDataset(val_images, val_labels)
        self.test_dataset = TensorDataset(test_images, test_labels)
        
        # 分析类别分布
        from collections import Counter
        train_distribution = Counter(train_labels.tolist())
        
        logger.info(f"虚拟数据集准备完成:")
        logger.info(f"  训练集: {len(train_labels):,} 样本")
        logger.info(f"  验证集: {len(val_labels):,} 样本")
        logger.info(f"  测试集: {len(test_labels):,} 样本")
        logger.info(f"  类别分布不平衡度: {max(train_distribution.values()) / min(train_distribution.values()):.2f}")
        
        return {
            'total_samples': len(all_labels),
            'train_samples': len(train_labels),
            'val_samples': len(val_labels),
            'test_samples': len(test_labels),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': dict(train_distribution)
        }
    
    def create_balanced_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建平衡的数据加载器"""
        logger.info("创建平衡数据加载器...")
        
        # 创建类别平衡策略
        balance_strategy = ClassBalanceStrategy('weighted_sampling')
        
        # 获取训练集标签
        train_labels = [self.train_dataset[i][1].item() for i in range(len(self.train_dataset))]
        
        # 创建加权采样器
        sampler = balance_strategy.create_weighted_sampler(
            train_labels, len(self.class_names)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=0,  # Windows兼容性
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info("平衡数据加载器创建完成")
        return train_loader, val_loader, test_loader
    
    def setup_advanced_training(self) -> Dict[str, Any]:
        """设置高级训练环境"""
        logger.info("设置高级训练环境...")
        
        # 更新配置中的类别数
        self.config.num_classes = len(self.class_names)
        
        # 创建训练器
        self.trainer = ModelTrainer(self.config)
        
        # 设置模型
        model = self.trainer.setup_model()
        
        # 使用Focal Loss处理类别不平衡
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.trainer.criterion = focal_loss.to(self.trainer.device)
        
        # 创建评估器
        self.evaluator = create_evaluator(
            class_names=self.class_names,
            device=str(self.trainer.device)
        )
        
        # 设置渐进式训练
        progressive_trainer = ProgressiveTrainer(
            model, 
            self.performance_tuner.hyperopt.create_default_search_space()
        )
        
        # 获取模型信息
        model_info = ModelFactory.get_model_info(model)
        
        setup_info = {
            'model_info': model_info,
            'device': str(self.trainer.device),
            'loss_function': 'FocalLoss',
            'balance_strategy': 'WeightedSampling',
            'progressive_training': True
        }
        
        logger.info(f"高级训练环境设置完成:")
        logger.info(f"  模型: {model_info['model_type']}")
        logger.info(f"  参数数量: {model_info['total_parameters']:,}")
        logger.info(f"  设备: {setup_info['device']}")
        logger.info(f"  损失函数: {setup_info['loss_function']}")
        
        return setup_info
    
    def train_with_optimization(self, 
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              enable_hyperopt: bool = False) -> Dict[str, Any]:
        """执行优化训练"""
        logger.info("开始优化训练...")
        
        if enable_hyperopt:
            # 超参数优化
            logger.info("执行超参数优化...")
            hyperopt = create_hyperparameter_optimizer('random', max_trials=10)
            
            hyperopt_results = hyperopt.optimize(
                self.config, train_loader, val_loader
            )
            
            # 使用最佳参数更新配置
            best_params = hyperopt_results['best_params']
            for key, value in best_params.items():
                setattr(self.config, key, value)
            
            # 重新创建训练器
            self.trainer = ModelTrainer(self.config)
            model = self.trainer.setup_model()
            
            logger.info(f"使用优化后的超参数: {best_params}")
        
        # 执行训练
        training_results = self.trainer.train(train_loader, val_loader)
        
        logger.info(f"优化训练完成:")
        logger.info(f"  最佳验证准确率: {training_results['best_val_acc']:.2f}%")
        logger.info(f"  最佳轮次: {training_results['best_epoch']}")
        logger.info(f"  总训练时间: {training_results['total_time']:.2f}秒")
        
        return training_results
    
    def evaluate_with_optimization(self, test_loader: DataLoader) -> Dict[str, Any]:
        """执行优化评估"""
        logger.info("开始优化评估...")
        
        # 加载最佳模型
        best_model_path = Path(self.config.save_dir) / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = self.trainer.load_checkpoint('best_model.pth')
            logger.info("已加载最佳模型权重")
        
        # 推理优化
        example_input = next(iter(test_loader))[0][:1].to(self.trainer.device)
        
        inference_opt = InferenceOptimizer()
        optimization_results = inference_opt.compare_optimizations(
            self.trainer.model, example_input
        )
        
        # 使用优化后的模型进行评估
        optimized_model = inference_opt.optimize_for_inference(
            self.trainer.model, example_input, 'basic'
        )
        
        # 评估模型
        metrics, predictions = self.evaluator.evaluate_model(
            optimized_model,
            test_loader,
            return_predictions=True
        )
        
        # 获取类别指标
        class_metrics = self.evaluator.get_class_metrics()
        
        # 保存评估报告
        self.evaluator.save_evaluation_report(
            metrics,
            save_dir=self.config.save_dir,
            model_name='optimized_plant_disease_model'
        )
        
        evaluation_results = {
            'overall_metrics': {
                'accuracy': metrics.accuracy,
                'f1_macro': metrics.f1_macro,
                'f1_weighted': metrics.f1_weighted,
                'precision_macro': metrics.precision_macro,
                'recall_macro': metrics.recall_macro
            },
            'top_k_accuracy': metrics.top_k_accuracy,
            'inference_optimization': optimization_results,
            'class_metrics_summary': {
                'best_class': max(class_metrics, key=lambda x: x.f1_score),
                'worst_class': min(class_metrics, key=lambda x: x.f1_score),
                'avg_f1': np.mean([cm.f1_score for cm in class_metrics])
            }
        }
        
        logger.info(f"优化评估完成:")
        logger.info(f"  测试准确率: {metrics.accuracy:.4f}")
        logger.info(f"  F1分数(宏): {metrics.f1_macro:.4f}")
        logger.info(f"  推理加速比: {optimization_results['speedup']['basic']:.2f}x")
        
        return evaluation_results
    
    def run_complete_optimized_training(self, enable_hyperopt: bool = False) -> Dict[str, Any]:
        """运行完整的优化训练流程"""
        logger.info("=" * 80)
        logger.info("开始优化版植物病害识别模型训练")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. 准备平衡数据集
            dataset_summary = self.prepare_datasets_with_balance()
            
            # 2. 创建平衡数据加载器
            train_loader, val_loader, test_loader = self.create_balanced_dataloaders()
            
            # 3. 设置高级训练环境
            setup_info = self.setup_advanced_training()
            
            # 4. 执行优化训练
            training_results = self.train_with_optimization(
                train_loader, val_loader, enable_hyperopt
            )
            
            # 5. 执行优化评估
            evaluation_results = self.evaluate_with_optimization(test_loader)
            
            # 6. 保存完整报告
            complete_results = {
                'dataset_summary': dataset_summary,
                'setup_info': setup_info,
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
            
            self.performance_tuner.save_optimization_report(
                complete_results,
                str(Path(self.config.save_dir) / 'complete_optimization_report.json')
            )
            
            total_time = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info("优化训练流程完成")
            logger.info(f"总耗时: {total_time:.2f}秒")
            logger.info(f"最终测试准确率: {evaluation_results['overall_metrics']['accuracy']:.4f}")
            logger.info(f"推理优化加速: {evaluation_results['inference_optimization']['speedup']['basic']:.2f}x")
            logger.info("=" * 80)
            
            return {
                'success': True,
                'total_time': total_time,
                'results': complete_results
            }
            
        except Exception as e:
            logger.error(f"优化训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化版植物病害识别模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--hyperopt', action='store_true', help='启用超参数优化')
    parser.add_argument('--gpu-config', action='store_true', help='使用GPU优化配置')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，请运行: pip install -r requirements.txt")
        return
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"🚀 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练")
    
    # 使用GPU配置
    config_path = args.config
    if args.gpu_config and not config_path:
        config_path = 'gpu_training_config.json'
        if not os.path.exists(config_path):
            print("❌ GPU配置文件不存在，请先运行: python check_gpu.py")
            return
    
    # 创建优化训练管理器
    training_manager = OptimizedTrainingManager(config_path)
    
    # 运行完整优化训练
    results = training_manager.run_complete_optimized_training(
        enable_hyperopt=args.hyperopt
    )
    
    if results['success']:
        print("\n🎉 优化训练成功完成!")
        final_results = results['results']['evaluation_results']['overall_metrics']
        print(f"📊 最终准确率: {final_results['accuracy']:.4f}")
        print(f"📊 F1分数: {final_results['f1_macro']:.4f}")
        print(f"⏱️  总耗时: {results['total_time']:.2f}秒")
        
        # 显示推理优化结果
        inference_results = results['results']['evaluation_results']['inference_optimization']
        print(f"🚀 推理优化:")
        print(f"   原始速度: {inference_results['original']['avg_inference_time_ms']:.1f}ms")
        print(f"   优化速度: {inference_results['basic_optimized']['avg_inference_time_ms']:.1f}ms")
        print(f"   加速比: {inference_results['speedup']['basic']:.2f}x")
    else:
        print(f"\n❌ 优化训练失败: {results['error']}")

if __name__ == "__main__":
    main()