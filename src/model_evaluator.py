#!/usr/bin/env python3
"""
模型评估和验证系统
实现ModelEvaluator类，支持准确率、F1分数、混淆矩阵和交叉验证
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, asdict
import warnings

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Subset
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve, auc
    )
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import pandas as pd
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """评估指标"""
    accuracy: float = 0.0
    precision_macro: float = 0.0
    precision_micro: float = 0.0
    precision_weighted: float = 0.0
    recall_macro: float = 0.0
    recall_micro: float = 0.0
    recall_weighted: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0
    auc_macro: float = 0.0
    auc_micro: float = 0.0
    top_k_accuracy: Dict[int, float] = None
    
    def __post_init__(self):
        if self.top_k_accuracy is None:
            self.top_k_accuracy = {}

@dataclass
class ClassMetrics:
    """单个类别的指标"""
    class_name: str
    class_id: int
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    support: int = 0
    auc: float = 0.0

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 class_names: Optional[List[str]] = None,
                 device: str = 'auto'):
        """
        初始化评估器
        
        Args:
            class_names: 类别名称列表
            device: 计算设备
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("需要安装必要依赖")
        
        self.class_names = class_names
        self.device = self._setup_device(device)
        
        # 评估结果存储
        self.last_predictions = None
        self.last_targets = None
        self.last_probabilities = None
        self.evaluation_history = []
        
        logger.info(f"ModelEvaluator初始化完成 - 设备: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def evaluate_model(self, 
                      model: nn.Module,
                      data_loader: DataLoader,
                      criterion: Optional[nn.Module] = None,
                      return_predictions: bool = True) -> Tuple[EvaluationMetrics, Optional[Dict[str, np.ndarray]]]:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            data_loader: 数据加载器
            criterion: 损失函数
            return_predictions: 是否返回预测结果
            
        Returns:
            (评估指标, 预测结果字典)
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc='Evaluating'):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                
                # 计算损失
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                # 获取预测和概率
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 转换为numpy数组
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        # 存储结果
        self.last_predictions = predictions
        self.last_targets = targets
        self.last_probabilities = probabilities
        
        # 计算指标
        metrics = self._calculate_metrics(targets, predictions, probabilities)
        
        # 计算平均损失
        if criterion is not None:
            avg_loss = total_loss / len(data_loader)
            logger.info(f"平均损失: {avg_loss:.4f}")
        
        # 准备返回结果
        results = None
        if return_predictions:
            results = {
                'predictions': predictions,
                'targets': targets,
                'probabilities': probabilities
            }
        
        return metrics, results
    
    def _calculate_metrics(self, 
                          targets: np.ndarray, 
                          predictions: np.ndarray, 
                          probabilities: np.ndarray) -> EvaluationMetrics:
        """计算评估指标"""
        
        # 基础指标
        accuracy = accuracy_score(targets, predictions)
        
        # 精确率、召回率、F1分数
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        precision_micro = precision_score(targets, predictions, average='micro', zero_division=0)
        precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
        
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        recall_micro = recall_score(targets, predictions, average='micro', zero_division=0)
        recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
        
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(targets, predictions, average='micro', zero_division=0)
        f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # AUC指标（多分类）
        auc_macro = 0.0
        auc_micro = 0.0
        
        try:
            # 计算多分类AUC
            n_classes = probabilities.shape[1]
            
            # 宏平均AUC
            auc_scores = []
            for i in range(n_classes):
                if len(np.unique(targets == i)) > 1:  # 确保类别存在
                    auc_score = roc_auc_score((targets == i).astype(int), probabilities[:, i])
                    auc_scores.append(auc_score)
            
            if auc_scores:
                auc_macro = np.mean(auc_scores)
            
            # 微平均AUC
            if n_classes > 2:
                # 对于多分类，使用one-vs-rest方式
                targets_onehot = np.eye(n_classes)[targets]
                auc_micro = roc_auc_score(targets_onehot, probabilities, average='micro', multi_class='ovr')
        
        except Exception as e:
            logger.warning(f"AUC计算失败: {e}")
        
        # Top-k准确率
        top_k_accuracy = {}
        for k in [1, 3, 5]:
            if k <= probabilities.shape[1]:
                top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
                top_k_acc = np.mean([targets[i] in top_k_preds[i] for i in range(len(targets))])
                top_k_accuracy[k] = top_k_acc
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision_macro=precision_macro,
            precision_micro=precision_micro,
            precision_weighted=precision_weighted,
            recall_macro=recall_macro,
            recall_micro=recall_micro,
            recall_weighted=recall_weighted,
            f1_macro=f1_macro,
            f1_micro=f1_micro,
            f1_weighted=f1_weighted,
            auc_macro=auc_macro,
            auc_micro=auc_micro,
            top_k_accuracy=top_k_accuracy
        )
    
    def get_class_metrics(self) -> List[ClassMetrics]:
        """获取每个类别的详细指标"""
        if self.last_predictions is None or self.last_targets is None:
            raise ValueError("请先运行evaluate_model")
        
        # 计算每个类别的指标
        precision_per_class = precision_score(
            self.last_targets, self.last_predictions, 
            average=None, zero_division=0
        )
        recall_per_class = recall_score(
            self.last_targets, self.last_predictions, 
            average=None, zero_division=0
        )
        f1_per_class = f1_score(
            self.last_targets, self.last_predictions, 
            average=None, zero_division=0
        )
        
        # 计算支持数（每个类别的样本数）
        unique_targets, support_counts = np.unique(self.last_targets, return_counts=True)
        support_dict = dict(zip(unique_targets, support_counts))
        
        # 计算每个类别的AUC
        auc_per_class = []
        n_classes = self.last_probabilities.shape[1]
        
        for i in range(n_classes):
            try:
                if len(np.unique(self.last_targets == i)) > 1:
                    auc_score = roc_auc_score(
                        (self.last_targets == i).astype(int), 
                        self.last_probabilities[:, i]
                    )
                    auc_per_class.append(auc_score)
                else:
                    auc_per_class.append(0.0)
            except:
                auc_per_class.append(0.0)
        
        # 创建类别指标列表
        class_metrics = []
        for i in range(len(precision_per_class)):
            class_name = self.class_names[i] if self.class_names else f'Class_{i}'
            
            metrics = ClassMetrics(
                class_name=class_name,
                class_id=i,
                precision=precision_per_class[i],
                recall=recall_per_class[i],
                f1_score=f1_per_class[i],
                support=support_dict.get(i, 0),
                auc=auc_per_class[i]
            )
            class_metrics.append(metrics)
        
        return class_metrics
    
    def plot_confusion_matrix(self, 
                            save_path: Optional[str] = None,
                            normalize: bool = True,
                            figsize: Tuple[int, int] = (12, 10)) -> None:
        """绘制混淆矩阵"""
        if self.last_predictions is None or self.last_targets is None:
            raise ValueError("请先运行evaluate_model")
        
        # 计算混淆矩阵
        cm = confusion_matrix(self.last_targets, self.last_predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # 绘制
        plt.figure(figsize=figsize)
        
        # 使用类别名称作为标签
        labels = self.class_names if self.class_names else [f'Class_{i}' for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, 
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """绘制ROC曲线"""
        if self.last_predictions is None or self.last_targets is None:
            raise ValueError("请先运行evaluate_model")
        
        n_classes = self.last_probabilities.shape[1]
        
        plt.figure(figsize=figsize)
        
        # 为每个类别绘制ROC曲线
        for i in range(min(n_classes, 10)):  # 最多显示10个类别
            # 二分类标签
            y_true = (self.last_targets == i).astype(int)
            y_score = self.last_probabilities[:, i]
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            # 绘制
            class_name = self.class_names[i] if self.class_names else f'Class {i}'
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC曲线已保存: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, 
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (12, 8)) -> None:
        """绘制精确率-召回率曲线"""
        if self.last_predictions is None or self.last_targets is None:
            raise ValueError("请先运行evaluate_model")
        
        n_classes = self.last_probabilities.shape[1]
        
        plt.figure(figsize=figsize)
        
        # 为每个类别绘制PR曲线
        for i in range(min(n_classes, 10)):  # 最多显示10个类别
            # 二分类标签
            y_true = (self.last_targets == i).astype(int)
            y_score = self.last_probabilities[:, i]
            
            # 计算PR曲线
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            
            # 绘制
            class_name = self.class_names[i] if self.class_names else f'Class {i}'
            plt.plot(recall, precision, linewidth=2,
                    label=f'{class_name} (AUC = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR曲线已保存: {save_path}")
        
        plt.show()
    
    def cross_validate(self, 
                      model_factory: Callable,
                      dataset: Dataset,
                      k_folds: int = 5,
                      batch_size: int = 32,
                      num_workers: int = 0) -> Dict[str, Any]:
        """
        K折交叉验证
        
        Args:
            model_factory: 模型工厂函数，返回新的模型实例
            dataset: 数据集
            k_folds: 折数
            batch_size: 批大小
            num_workers: 工作进程数
            
        Returns:
            交叉验证结果
        """
        logger.info(f"开始{k_folds}折交叉验证...")
        
        # 获取所有标签用于分层采样
        all_targets = []
        for i in range(len(dataset)):
            _, target = dataset[i]
            all_targets.append(target)
        
        all_targets = np.array(all_targets)
        
        # 创建分层K折
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_targets)), all_targets)):
            logger.info(f"执行第 {fold + 1}/{k_folds} 折...")
            
            # 创建数据子集
            val_subset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers)
            
            # 创建新模型
            model = model_factory()
            model.to(self.device)
            
            # 评估模型（这里假设模型已经训练好了）
            # 在实际应用中，你需要在这里训练模型
            metrics, _ = self.evaluate_model(model, val_loader, return_predictions=False)
            
            fold_result = {
                'fold': fold + 1,
                'metrics': asdict(metrics),
                'val_size': len(val_idx)
            }
            fold_results.append(fold_result)
            
            logger.info(f"第 {fold + 1} 折结果 - 准确率: {metrics.accuracy:.4f}, F1: {metrics.f1_macro:.4f}")
        
        # 计算平均结果
        avg_metrics = {}
        for key in fold_results[0]['metrics'].keys():
            if key != 'top_k_accuracy':
                values = [result['metrics'][key] for result in fold_results]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        
        # 处理top_k_accuracy
        if 'top_k_accuracy' in fold_results[0]['metrics']:
            avg_top_k = {}
            for k in fold_results[0]['metrics']['top_k_accuracy'].keys():
                values = [result['metrics']['top_k_accuracy'][k] for result in fold_results]
                avg_top_k[k] = np.mean(values)
            avg_metrics['top_k_accuracy'] = avg_top_k
        
        cv_results = {
            'fold_results': fold_results,
            'average_metrics': avg_metrics,
            'k_folds': k_folds,
            'total_samples': len(dataset)
        }
        
        logger.info(f"交叉验证完成 - 平均准确率: {avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        
        return cv_results
    
    def generate_classification_report(self) -> str:
        """生成分类报告"""
        if self.last_predictions is None or self.last_targets is None:
            raise ValueError("请先运行evaluate_model")
        
        target_names = self.class_names if self.class_names else None
        
        report = classification_report(
            self.last_targets, 
            self.last_predictions,
            target_names=target_names,
            digits=4
        )
        
        return report
    
    def save_evaluation_report(self, 
                             metrics: EvaluationMetrics,
                             save_dir: str,
                             model_name: str = 'model') -> None:
        """保存评估报告"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标
        metrics_dict = asdict(metrics)
        with open(save_dir / f'{model_name}_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # 保存分类报告
        if self.last_predictions is not None:
            report = self.generate_classification_report()
            with open(save_dir / f'{model_name}_classification_report.txt', 'w') as f:
                f.write(report)
        
        # 保存类别指标
        if self.last_predictions is not None:
            class_metrics = self.get_class_metrics()
            class_metrics_dict = [asdict(cm) for cm in class_metrics]
            with open(save_dir / f'{model_name}_class_metrics.json', 'w') as f:
                json.dump(class_metrics_dict, f, indent=2)
        
        # 绘制并保存图表
        if self.last_predictions is not None:
            self.plot_confusion_matrix(save_path=save_dir / f'{model_name}_confusion_matrix.png')
            self.plot_roc_curves(save_path=save_dir / f'{model_name}_roc_curves.png')
            self.plot_precision_recall_curves(save_path=save_dir / f'{model_name}_pr_curves.png')
        
        logger.info(f"评估报告已保存到: {save_dir}")

# 便捷函数
def create_evaluator(class_names: Optional[List[str]] = None, 
                    device: str = 'auto') -> ModelEvaluator:
    """创建模型评估器"""
    return ModelEvaluator(class_names=class_names, device=device)

def evaluate_model_simple(model: nn.Module, 
                         data_loader: DataLoader,
                         class_names: Optional[List[str]] = None) -> EvaluationMetrics:
    """简单模型评估"""
    evaluator = create_evaluator(class_names=class_names)
    metrics, _ = evaluator.evaluate_model(model, data_loader, return_predictions=False)
    return metrics

if __name__ == "__main__":
    # 测试模型评估系统
    print("🧪 模型评估系统测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试评估器创建
        print("📋 测试评估器创建...")
        class_names = [f'Class_{i}' for i in range(10)]
        evaluator = create_evaluator(class_names=class_names)
        
        print(f"✅ 评估器创建成功:")
        print(f"   设备: {evaluator.device}")
        print(f"   类别数: {len(class_names)}")
        
        # 创建模拟数据
        print(f"\n🔍 测试指标计算...")
        
        # 模拟预测结果
        n_samples = 1000
        n_classes = 10
        
        targets = np.random.randint(0, n_classes, n_samples)
        predictions = np.random.randint(0, n_classes, n_samples)
        probabilities = np.random.rand(n_samples, n_classes)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # 设置模拟结果
        evaluator.last_targets = targets
        evaluator.last_predictions = predictions
        evaluator.last_probabilities = probabilities
        
        # 计算指标
        metrics = evaluator._calculate_metrics(targets, predictions, probabilities)
        
        print(f"✅ 指标计算完成:")
        print(f"   准确率: {metrics.accuracy:.4f}")
        print(f"   F1分数(宏): {metrics.f1_macro:.4f}")
        print(f"   F1分数(微): {metrics.f1_micro:.4f}")
        print(f"   AUC(宏): {metrics.auc_macro:.4f}")
        print(f"   Top-1准确率: {metrics.top_k_accuracy.get(1, 0):.4f}")
        print(f"   Top-3准确率: {metrics.top_k_accuracy.get(3, 0):.4f}")
        
        # 测试类别指标
        print(f"\n📊 测试类别指标...")
        class_metrics = evaluator.get_class_metrics()
        
        print(f"✅ 类别指标计算完成:")
        print(f"   类别数: {len(class_metrics)}")
        print(f"   前3个类别:")
        for i, cm in enumerate(class_metrics[:3]):
            print(f"     {cm.class_name}: P={cm.precision:.3f}, R={cm.recall:.3f}, F1={cm.f1_score:.3f}")
        
        # 测试分类报告
        print(f"\n📄 测试分类报告...")
        report = evaluator.generate_classification_report()
        print(f"✅ 分类报告生成完成:")
        print(f"   报告长度: {len(report)} 字符")
        print(f"   前200字符: {report[:200]}...")
        
        # 测试评估指标数据类
        print(f"\n🏷️ 测试评估指标数据类...")
        metrics_dict = asdict(metrics)
        print(f"✅ 指标序列化完成:")
        print(f"   指标数量: {len(metrics_dict)}")
        print(f"   主要指标: accuracy, f1_macro, precision_macro")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 模型评估系统测试完成")