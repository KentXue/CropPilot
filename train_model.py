#!/usr/bin/env python3
"""
植物病害识别模型训练脚本
整合所有组件，执行完整的模型训练流程
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
    from torch.utils.data import DataLoader, Dataset, random_split
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
from src.dataset_manager import DatasetManager
from src.plantvillage_loader import PlantVillageLoader
from src.baidu_dataset_loader import BaiduDatasetLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """植物病害数据集包装器"""
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 class_names: List[str],
                 preprocessor: PlantDiseasePreprocessor):
        """
        初始化数据集
        
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            class_names: 类别名称列表
            preprocessor: 图像预处理器
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.preprocessor = preprocessor
        
        assert len(image_paths) == len(labels), "图像数量与标签数量不匹配"
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 预处理图像
            image = self.preprocessor.preprocess_image(image_path)
            return image, label
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            # 返回零张量作为备用
            zero_image = torch.zeros(3, *self.preprocessor.input_size)
            return zero_image, label

class TrainingManager:
    """训练管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化训练管理器
        
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
        
        # 数据集信息
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = []
        
        logger.info("训练管理器初始化完成")
    
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
            # 使用默认配置
            config = create_default_config(
                num_epochs=50,
                batch_size=32,
                learning_rate=0.001,
                model_name='efficientnet-b4',
                pretrained=True,
                early_stopping=True,
                patience=10,
                save_dir='checkpoints/plant_disease_model'
            )
            logger.info("使用默认训练配置")
        
        return config
    
    def prepare_datasets(self) -> Dict[str, Any]:
        """准备训练数据集"""
        logger.info("开始准备数据集...")
        
        # 加载PlantVillage数据集
        plantvillage_loader = PlantVillageLoader()
        plantvillage_data = plantvillage_loader.load_dataset()
        
        if not plantvillage_data['success']:
            raise RuntimeError(f"PlantVillage数据集加载失败: {plantvillage_data['message']}")
        
        # 获取数据集信息
        dataset_info = plantvillage_data['dataset_info']
        self.class_names = list(dataset_info['class_mapping'].values())
        
        logger.info(f"数据集加载完成:")
        logger.info(f"  总样本数: {dataset_info['total_samples']:,}")
        logger.info(f"  类别数: {dataset_info['num_classes']}")
        logger.info(f"  图像尺寸范围: {dataset_info['image_size_stats']}")
        
        # 准备图像路径和标签
        image_paths = []
        labels = []
        
        for class_name, class_data in plantvillage_data['class_data'].items():
            class_id = list(dataset_info['class_mapping'].keys())[
                list(dataset_info['class_mapping'].values()).index(class_name)
            ]
            
            for img_path in class_data['image_paths']:
                image_paths.append(img_path)
                labels.append(class_id)
        
        # 数据集分割
        total_size = len(image_paths)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        # 随机分割（保持类别平衡）
        from sklearn.model_selection import train_test_split
        
        # 先分出训练集
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, 
            test_size=(val_size + test_size),
            stratify=labels,
            random_state=42
        )
        
        # 再分出验证集和测试集
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=test_size,
            stratify=temp_labels,
            random_state=42
        )
        
        # 创建预处理器
        train_preprocessor = PlantDiseasePreprocessor(
            input_size=(224, 224),
            mode=PreprocessingMode.TRAINING
        )
        val_preprocessor = PlantDiseasePreprocessor(
            input_size=(224, 224),
            mode=PreprocessingMode.VALIDATION
        )
        
        # 创建数据集对象
        self.train_dataset = PlantDiseaseDataset(
            train_paths, train_labels, self.class_names, train_preprocessor
        )
        self.val_dataset = PlantDiseaseDataset(
            val_paths, val_labels, self.class_names, val_preprocessor
        )
        self.test_dataset = PlantDiseaseDataset(
            test_paths, test_labels, self.class_names, val_preprocessor
        )
        
        dataset_summary = {
            'total_samples': total_size,
            'train_samples': len(train_paths),
            'val_samples': len(val_paths),
            'test_samples': len(test_paths),
            'num_classes': len(self.class_names),
            'class_names': self.class_names
        }
        
        logger.info(f"数据集分割完成:")
        logger.info(f"  训练集: {dataset_summary['train_samples']:,} 样本")
        logger.info(f"  验证集: {dataset_summary['val_samples']:,} 样本")
        logger.info(f"  测试集: {dataset_summary['test_samples']:,} 样本")
        
        return dataset_summary
    
    def setup_training(self) -> Dict[str, Any]:
        """设置训练环境"""
        logger.info("设置训练环境...")
        
        # 更新配置中的类别数
        self.config.num_classes = len(self.class_names)
        
        # 创建训练器
        self.trainer = ModelTrainer(self.config)
        
        # 设置模型
        model = self.trainer.setup_model()
        
        # 创建评估器
        self.evaluator = create_evaluator(
            class_names=self.class_names,
            device=str(self.trainer.device)
        )
        
        # 获取模型信息
        model_info = ModelFactory.get_model_info(model)
        
        setup_info = {
            'model_info': model_info,
            'device': str(self.trainer.device),
            'optimizer': type(self.trainer.optimizer).__name__,
            'scheduler': type(self.trainer.scheduler).__name__,
            'criterion': type(self.trainer.criterion).__name__
        }
        
        logger.info(f"训练环境设置完成:")
        logger.info(f"  模型: {model_info['model_type']}")
        logger.info(f"  参数数量: {model_info['total_parameters']:,}")
        logger.info(f"  设备: {setup_info['device']}")
        logger.info(f"  优化器: {setup_info['optimizer']}")
        
        return setup_info
    
    def train_model(self) -> Dict[str, Any]:
        """执行模型训练"""
        logger.info("开始模型训练...")
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 执行训练
        training_results = self.trainer.train(train_loader, val_loader)
        
        logger.info(f"训练完成:")
        logger.info(f"  最佳验证准确率: {training_results['best_val_acc']:.2f}%")
        logger.info(f"  最佳轮次: {training_results['best_epoch']}")
        logger.info(f"  总训练时间: {training_results['total_time']:.2f}秒")
        
        return training_results
    
    def evaluate_model(self) -> Dict[str, Any]:
        """评估训练好的模型"""
        logger.info("开始模型评估...")
        
        # 加载最佳模型
        best_model_path = Path(self.config.save_dir) / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = self.trainer.load_checkpoint('best_model.pth')
            logger.info("已加载最佳模型权重")
        
        # 创建测试数据加载器
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # 评估模型
        metrics, predictions = self.evaluator.evaluate_model(
            self.trainer.model,
            test_loader,
            criterion=self.trainer.criterion,
            return_predictions=True
        )
        
        # 获取类别指标
        class_metrics = self.evaluator.get_class_metrics()
        
        # 生成分类报告
        classification_report = self.evaluator.generate_classification_report()
        
        # 保存评估报告
        self.evaluator.save_evaluation_report(
            metrics,
            save_dir=self.config.save_dir,
            model_name='plant_disease_efficientnet'
        )
        
        evaluation_results = {
            'overall_metrics': {
                'accuracy': metrics.accuracy,
                'f1_macro': metrics.f1_macro,
                'f1_weighted': metrics.f1_weighted,
                'precision_macro': metrics.precision_macro,
                'recall_macro': metrics.recall_macro,
                'auc_macro': metrics.auc_macro
            },
            'top_k_accuracy': metrics.top_k_accuracy,
            'class_metrics': [
                {
                    'class_name': cm.class_name,
                    'precision': cm.precision,
                    'recall': cm.recall,
                    'f1_score': cm.f1_score,
                    'support': cm.support
                }
                for cm in class_metrics
            ],
            'classification_report': classification_report
        }
        
        logger.info(f"模型评估完成:")
        logger.info(f"  测试准确率: {metrics.accuracy:.4f}")
        logger.info(f"  F1分数(宏): {metrics.f1_macro:.4f}")
        logger.info(f"  F1分数(加权): {metrics.f1_weighted:.4f}")
        logger.info(f"  Top-3准确率: {metrics.top_k_accuracy.get(3, 0):.4f}")
        
        return evaluation_results
    
    def save_training_summary(self, 
                            dataset_summary: Dict[str, Any],
                            setup_info: Dict[str, Any],
                            training_results: Dict[str, Any],
                            evaluation_results: Dict[str, Any]):
        """保存训练总结"""
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'num_epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'model_name': self.config.model_name,
                'optimizer_type': self.config.optimizer_type,
                'scheduler_type': self.config.scheduler_type
            },
            'dataset_summary': dataset_summary,
            'setup_info': setup_info,
            'training_results': {
                'best_val_acc': training_results['best_val_acc'],
                'best_epoch': training_results['best_epoch'],
                'total_time': training_results['total_time']
            },
            'evaluation_results': evaluation_results
        }
        
        # 保存到文件
        summary_path = Path(self.config.save_dir) / 'training_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练总结已保存: {summary_path}")
    
    def run_complete_training(self) -> Dict[str, Any]:
        """运行完整的训练流程"""
        logger.info("=" * 60)
        logger.info("开始植物病害识别模型训练")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 准备数据集
            dataset_summary = self.prepare_datasets()
            
            # 2. 设置训练环境
            setup_info = self.setup_training()
            
            # 3. 执行训练
            training_results = self.train_model()
            
            # 4. 评估模型
            evaluation_results = self.evaluate_model()
            
            # 5. 保存总结
            self.save_training_summary(
                dataset_summary, setup_info, 
                training_results, evaluation_results
            )
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("训练流程完成")
            logger.info(f"总耗时: {total_time:.2f}秒")
            logger.info(f"最终测试准确率: {evaluation_results['overall_metrics']['accuracy']:.4f}")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'total_time': total_time,
                'dataset_summary': dataset_summary,
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

def create_training_config_file(config_path: str):
    """创建训练配置文件"""
    config = {
        "num_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "model_type": "efficientnet",
        "model_name": "efficientnet-b4",
        "num_classes": 38,
        "pretrained": True,
        "optimizer_type": "adamw",
        "scheduler_type": "cosine",
        "early_stopping": True,
        "patience": 10,
        "min_delta": 0.001,
        "save_dir": "checkpoints/plant_disease_model",
        "mixed_precision": True,
        "gradient_clip_norm": 1.0,
        "label_smoothing": 0.1
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"训练配置文件已创建: {config_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='植物病害识别模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--create-config', type=str, help='创建配置文件')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_training_config_file(args.create_config)
        return
    
    # 检查依赖
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，请运行: pip install -r requirements.txt")
        return
    
    # 创建训练管理器
    training_manager = TrainingManager(args.config)
    
    # 运行完整训练
    results = training_manager.run_complete_training()
    
    if results['success']:
        print("\n🎉 训练成功完成!")
        print(f"📊 最终准确率: {results['evaluation_results']['overall_metrics']['accuracy']:.4f}")
        print(f"⏱️  总耗时: {results['total_time']:.2f}秒")
    else:
        print(f"\n❌ 训练失败: {results['error']}")

if __name__ == "__main__":
    main()