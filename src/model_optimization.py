#!/usr/bin/env python3
"""
模型性能调优模块
实现超参数搜索、模型集成、推理优化等高级性能调优功能
"""

import os
import sys
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, asdict
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    from sklearn.model_selection import ParameterGrid
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

from src.model_architecture import ModelFactory, create_plant_disease_model
from src.model_trainer import ModelTrainer, TrainingConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HyperparameterConfig:
    """超参数配置"""
    learning_rates: List[float]
    batch_sizes: List[int]
    weight_decays: List[float]
    dropout_rates: List[float]
    optimizers: List[str]
    schedulers: List[str]
    
    def to_grid(self) -> List[Dict[str, Any]]:
        """转换为参数网格"""
        param_dict = {
            'learning_rate': self.learning_rates,
            'batch_size': self.batch_sizes,
            'weight_decay': self.weight_decays,
            'dropout_rate': self.dropout_rates,
            'optimizer_type': self.optimizers,
            'scheduler_type': self.schedulers
        }
        return list(ParameterGrid(param_dict))

class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self, 
                 search_strategy: str = 'grid',
                 max_trials: int = 50,
                 early_stopping_patience: int = 3):
        """
        初始化超参数优化器
        
        Args:
            search_strategy: 搜索策略 ('grid', 'random', 'bayesian')
            max_trials: 最大试验次数
            early_stopping_patience: 早停耐心值
        """
        self.search_strategy = search_strategy
        self.max_trials = max_trials
        self.early_stopping_patience = early_stopping_patience
        self.trial_results = []
        
    def create_default_search_space(self) -> HyperparameterConfig:
        """创建默认搜索空间"""
        return HyperparameterConfig(
            learning_rates=[0.0001, 0.001, 0.01],
            batch_sizes=[8, 16, 32],
            weight_decays=[1e-5, 1e-4, 1e-3],
            dropout_rates=[0.2, 0.3, 0.4],
            optimizers=['adam', 'adamw'],
            schedulers=['cosine', 'step']
        )
    
    def grid_search(self, config: HyperparameterConfig) -> List[Dict[str, Any]]:
        """网格搜索"""
        param_grid = config.to_grid()
        
        # 限制试验次数
        if len(param_grid) > self.max_trials:
            param_grid = random.sample(param_grid, self.max_trials)
            logger.info(f"参数组合过多，随机选择 {self.max_trials} 个组合进行搜索")
        
        return param_grid
    
    def random_search(self, config: HyperparameterConfig) -> List[Dict[str, Any]]:
        """随机搜索"""
        param_combinations = []
        
        for _ in range(self.max_trials):
            params = {
                'learning_rate': random.choice(config.learning_rates),
                'batch_size': random.choice(config.batch_sizes),
                'weight_decay': random.choice(config.weight_decays),
                'dropout_rate': random.choice(config.dropout_rates),
                'optimizer_type': random.choice(config.optimizers),
                'scheduler_type': random.choice(config.schedulers)
            }
            param_combinations.append(params)
        
        return param_combinations
    
    def optimize(self, 
                 base_config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 search_config: Optional[HyperparameterConfig] = None) -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            base_config: 基础训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            search_config: 搜索配置
            
        Returns:
            优化结果
        """
        if search_config is None:
            search_config = self.create_default_search_space()
        
        # 获取参数组合
        if self.search_strategy == 'grid':
            param_combinations = self.grid_search(search_config)
        elif self.search_strategy == 'random':
            param_combinations = self.random_search(search_config)
        else:
            raise ValueError(f"不支持的搜索策略: {self.search_strategy}")
        
        logger.info(f"开始超参数优化，共 {len(param_combinations)} 个组合")
        
        best_score = 0.0
        best_params = None
        no_improvement_count = 0
        
        for i, params in enumerate(param_combinations):
            logger.info(f"试验 {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # 创建训练配置
                trial_config = TrainingConfig()
                for key, value in asdict(base_config).items():
                    setattr(trial_config, key, value)
                
                # 更新超参数
                for key, value in params.items():
                    setattr(trial_config, key, value)
                
                # 短训练用于评估
                trial_config.num_epochs = 5
                trial_config.save_dir = f"hyperopt_trial_{i}"
                
                # 训练模型
                trainer = ModelTrainer(trial_config)
                model = trainer.setup_model()
                
                training_results = trainer.train(train_loader, val_loader)
                
                # 记录结果
                trial_result = {
                    'trial_id': i,
                    'params': params,
                    'val_acc': training_results['best_val_acc'],
                    'training_time': training_results['total_time']
                }
                
                self.trial_results.append(trial_result)
                
                # 检查是否是最佳结果
                if training_results['best_val_acc'] > best_score:
                    best_score = training_results['best_val_acc']
                    best_params = params
                    no_improvement_count = 0
                    logger.info(f"新的最佳结果: {best_score:.4f}")
                else:
                    no_improvement_count += 1
                
                # 早停检查
                if no_improvement_count >= self.early_stopping_patience:
                    logger.info(f"连续 {self.early_stopping_patience} 次无改善，提前停止")
                    break
                
                # 清理
                del trainer, model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 删除临时文件
                import shutil
                if os.path.exists(trial_config.save_dir):
                    shutil.rmtree(trial_config.save_dir)
                
            except Exception as e:
                logger.error(f"试验 {i+1} 失败: {e}")
                continue
        
        # 整理结果
        optimization_results = {
            'best_params': best_params,
            'best_score': best_score,
            'total_trials': len(self.trial_results),
            'trial_results': self.trial_results
        }
        
        logger.info(f"超参数优化完成，最佳准确率: {best_score:.4f}")
        logger.info(f"最佳参数: {best_params}")
        
        return optimization_results

class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self, ensemble_method: str = 'voting'):
        """
        初始化模型集成器
        
        Args:
            ensemble_method: 集成方法 ('voting', 'weighted', 'stacking')
        """
        self.ensemble_method = ensemble_method
        self.models = []
        self.weights = []
        
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """添加模型到集成"""
        self.models.append(model)
        self.weights.append(weight)
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """集成预测"""
        if not self.models:
            raise ValueError("没有可用的模型")
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(torch.softmax(pred, dim=1))
        
        # 集成预测
        if self.ensemble_method == 'voting':
            # 简单平均
            ensemble_pred = torch.stack(predictions).mean(dim=0)
        elif self.ensemble_method == 'weighted':
            # 加权平均
            weights = torch.tensor(self.weights, device=x.device)
            weights = weights / weights.sum()
            
            weighted_preds = []
            for i, pred in enumerate(predictions):
                weighted_preds.append(pred * weights[i])
            
            ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
        
        return ensemble_pred
    
    def evaluate_ensemble(self, data_loader: DataLoader) -> Dict[str, float]:
        """评估集成模型"""
        correct = 0
        total = 0
        
        for data, targets in data_loader:
            predictions = self.predict(data)
            predicted = predictions.argmax(dim=1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = 100.0 * correct / total
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

class InferenceOptimizer:
    """推理优化器"""
    
    def __init__(self):
        """初始化推理优化器"""
        self.optimized_models = {}
        
    def optimize_for_inference(self, model: nn.Module, 
                             example_input: torch.Tensor,
                             optimization_level: str = 'basic') -> nn.Module:
        """
        优化模型用于推理
        
        Args:
            model: 要优化的模型
            example_input: 示例输入
            optimization_level: 优化级别 ('basic', 'advanced')
            
        Returns:
            优化后的模型
        """
        model.eval()
        
        if optimization_level == 'basic':
            # 基础优化：JIT编译
            try:
                optimized_model = torch.jit.trace(model, example_input)
                logger.info("JIT编译优化完成")
                return optimized_model
            except Exception as e:
                logger.warning(f"JIT编译失败: {e}")
                return model
        
        elif optimization_level == 'advanced':
            # 高级优化：量化 + JIT
            try:
                # 动态量化
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
                
                # JIT编译
                optimized_model = torch.jit.trace(quantized_model, example_input)
                
                logger.info("量化 + JIT编译优化完成")
                return optimized_model
            except Exception as e:
                logger.warning(f"高级优化失败: {e}")
                return self.optimize_for_inference(model, example_input, 'basic')
        
        return model
    
    def benchmark_inference(self, model: nn.Module, 
                          test_input: torch.Tensor,
                          num_runs: int = 100) -> Dict[str, float]:
        """推理性能基准测试"""
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # 同步GPU
        if test_input.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 性能测试
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        if test_input.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = test_input.size(0) / avg_time
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_images_per_sec': throughput,
            'total_time_sec': total_time
        }
    
    def compare_optimizations(self, original_model: nn.Module,
                            example_input: torch.Tensor) -> Dict[str, Any]:
        """比较不同优化方法的性能"""
        results = {}
        
        # 原始模型
        original_benchmark = self.benchmark_inference(original_model, example_input)
        results['original'] = original_benchmark
        
        # 基础优化
        basic_optimized = self.optimize_for_inference(original_model, example_input, 'basic')
        basic_benchmark = self.benchmark_inference(basic_optimized, example_input)
        results['basic_optimized'] = basic_benchmark
        
        # 高级优化
        advanced_optimized = self.optimize_for_inference(original_model, example_input, 'advanced')
        advanced_benchmark = self.benchmark_inference(advanced_optimized, example_input)
        results['advanced_optimized'] = advanced_benchmark
        
        # 计算加速比
        results['speedup'] = {
            'basic': original_benchmark['avg_inference_time_ms'] / basic_benchmark['avg_inference_time_ms'],
            'advanced': original_benchmark['avg_inference_time_ms'] / advanced_benchmark['avg_inference_time_ms']
        }
        
        return results

class PerformanceTuner:
    """性能调优器"""
    
    def __init__(self):
        """初始化性能调优器"""
        self.hyperopt = HyperparameterOptimizer()
        self.ensemble = ModelEnsemble()
        self.inference_opt = InferenceOptimizer()
        
    def full_optimization_pipeline(self, 
                                 base_config: TrainingConfig,
                                 train_loader: DataLoader,
                                 val_loader: DataLoader,
                                 test_loader: DataLoader) -> Dict[str, Any]:
        """完整的优化流程"""
        results = {}
        
        logger.info("开始完整优化流程...")
        
        # 1. 超参数优化
        logger.info("步骤1: 超参数优化")
        hyperopt_results = self.hyperopt.optimize(base_config, train_loader, val_loader)
        results['hyperparameter_optimization'] = hyperopt_results
        
        # 2. 使用最佳参数训练最终模型
        logger.info("步骤2: 训练最终模型")
        best_params = hyperopt_results['best_params']
        
        final_config = TrainingConfig()
        for key, value in asdict(base_config).items():
            setattr(final_config, key, value)
        
        for key, value in best_params.items():
            setattr(final_config, key, value)
        
        final_config.num_epochs = base_config.num_epochs  # 恢复完整训练轮数
        final_config.save_dir = "optimized_model"
        
        trainer = ModelTrainer(final_config)
        model = trainer.setup_model()
        training_results = trainer.train(train_loader, val_loader)
        
        results['final_training'] = training_results
        
        # 3. 推理优化
        logger.info("步骤3: 推理优化")
        example_input = next(iter(test_loader))[0][:1]  # 单个样本
        
        inference_results = self.inference_opt.compare_optimizations(model, example_input)
        results['inference_optimization'] = inference_results
        
        # 4. 模型评估
        logger.info("步骤4: 最终评估")
        from src.model_evaluator import create_evaluator
        
        evaluator = create_evaluator()
        metrics, _ = evaluator.evaluate_model(model, test_loader, return_predictions=False)
        
        results['final_evaluation'] = {
            'accuracy': metrics.accuracy,
            'f1_macro': metrics.f1_macro,
            'f1_weighted': metrics.f1_weighted
        }
        
        logger.info("完整优化流程完成")
        return results
    
    def save_optimization_report(self, results: Dict[str, Any], save_path: str):
        """保存优化报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_results': results,
            'summary': {
                'best_hyperparams': results.get('hyperparameter_optimization', {}).get('best_params'),
                'final_accuracy': results.get('final_evaluation', {}).get('accuracy'),
                'inference_speedup': results.get('inference_optimization', {}).get('speedup'),
                'training_time': results.get('final_training', {}).get('total_time')
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化报告已保存: {save_path}")

# 便捷函数
def create_hyperparameter_optimizer(strategy: str = 'random', max_trials: int = 20) -> HyperparameterOptimizer:
    """创建超参数优化器"""
    return HyperparameterOptimizer(search_strategy=strategy, max_trials=max_trials)

def create_model_ensemble(models: List[nn.Module], method: str = 'voting') -> ModelEnsemble:
    """创建模型集成"""
    ensemble = ModelEnsemble(ensemble_method=method)
    for model in models:
        ensemble.add_model(model)
    return ensemble

if __name__ == "__main__":
    # 测试模型性能调优
    print("🧪 模型性能调优测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试超参数配置
        print("⚙️ 测试超参数配置...")
        config = HyperparameterConfig(
            learning_rates=[0.001, 0.01],
            batch_sizes=[8, 16],
            weight_decays=[1e-4, 1e-3],
            dropout_rates=[0.2, 0.3],
            optimizers=['adam', 'adamw'],
            schedulers=['cosine', 'step']
        )
        
        param_grid = config.to_grid()
        print(f"✅ 参数网格大小: {len(param_grid)}")
        print(f"   示例参数: {param_grid[0]}")
        
        # 测试超参数优化器
        print(f"\n🔍 测试超参数优化器...")
        optimizer = HyperparameterOptimizer(search_strategy='random', max_trials=5)
        
        default_config = optimizer.create_default_search_space()
        print(f"✅ 默认搜索空间创建完成")
        print(f"   学习率范围: {default_config.learning_rates}")
        print(f"   批大小范围: {default_config.batch_sizes}")
        
        # 测试模型集成
        print(f"\n🤝 测试模型集成...")
        ensemble = ModelEnsemble(ensemble_method='voting')
        
        # 创建虚拟模型
        model1 = create_plant_disease_model('efficientnet', pretrained=False)
        model2 = create_plant_disease_model('efficientnet', pretrained=False)
        
        ensemble.add_model(model1, weight=1.0)
        ensemble.add_model(model2, weight=1.0)
        
        # 测试预测
        test_input = torch.randn(2, 3, 224, 224)
        ensemble_pred = ensemble.predict(test_input)
        
        print(f"✅ 集成预测完成")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {ensemble_pred.shape}")
        
        # 测试推理优化
        print(f"\n🚀 测试推理优化...")
        inference_opt = InferenceOptimizer()
        
        # 基准测试
        benchmark_results = inference_opt.benchmark_inference(model1, test_input, num_runs=10)
        
        print(f"✅ 推理基准测试完成:")
        print(f"   平均推理时间: {benchmark_results['avg_inference_time_ms']:.2f}ms")
        print(f"   吞吐量: {benchmark_results['throughput_images_per_sec']:.1f} images/sec")
        
        # 测试JIT优化
        try:
            optimized_model = inference_opt.optimize_for_inference(model1, test_input, 'basic')
            print(f"✅ JIT优化完成")
        except Exception as e:
            print(f"⚠️  JIT优化跳过: {e}")
        
        # 测试性能调优器
        print(f"\n🎯 测试性能调优器...")
        tuner = PerformanceTuner()
        
        print(f"✅ 性能调优器创建完成")
        print(f"   包含组件: 超参数优化、模型集成、推理优化")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 模型性能调优测试完成")