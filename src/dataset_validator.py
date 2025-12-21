#!/usr/bin/env python3
"""
数据集验证工具
验证数据集完整性和质量
"""

import os
import sys
from typing import Dict, List, Any, Tuple
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_manager import get_dataset_manager, DatasetManager
from src.dataset_config import get_dataset_config

logger = logging.getLogger(__name__)

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self):
        self.config = get_dataset_config()
        self.manager = get_dataset_manager()
        
    def validate_all_datasets(self) -> Dict[str, Any]:
        """验证所有数据集"""
        results = {
            'overall_status': 'unknown',
            'dataset_validations': {},
            'summary': {},
            'recommendations': []
        }
        
        # 验证各个数据集
        dataset_names = ['color', 'grayscale', 'segmented', 'baidu']
        
        for dataset_name in dataset_names:
            try:
                validation_result = self.validate_dataset(dataset_name)
                results['dataset_validations'][dataset_name] = validation_result
            except Exception as e:
                results['dataset_validations'][dataset_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # 生成总结和建议
        results['summary'] = self._generate_summary(results['dataset_validations'])
        results['recommendations'] = self._generate_recommendations(results['dataset_validations'])
        results['overall_status'] = self._determine_overall_status(results['dataset_validations'])
        
        return results
    
    def validate_dataset(self, dataset_name: str, max_samples: int = 100) -> Dict[str, Any]:
        """
        验证单个数据集
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大验证样本数
            
        Returns:
            验证结果字典
        """
        result = {
            'dataset_name': dataset_name,
            'status': 'unknown',
            'path_exists': False,
            'loadable': False,
            'sample_count': 0,
            'class_count': 0,
            'class_distribution': {},
            'issues': [],
            'warnings': []
        }
        
        try:
            # 检查路径存在性
            if dataset_name in ['color', 'grayscale', 'segmented']:
                config = self.config.plantvillage_datasets[dataset_name]
            elif dataset_name == 'baidu':
                config = self.config.baidu_dataset
            else:
                raise ValueError(f"未知数据集: {dataset_name}")
            
            result['path_exists'] = os.path.exists(config.path)
            
            if not result['path_exists']:
                result['status'] = 'path_not_found'
                result['issues'].append(f"数据集路径不存在: {config.path}")
                return result
            
            # 尝试加载数据集
            dataset = self.manager.load_dataset(
                dataset_name, 
                max_samples_per_class=max_samples // 38 if max_samples else None
            )
            
            result['loadable'] = True
            result['sample_count'] = len(dataset)
            result['class_count'] = len(dataset.classes)
            result['class_distribution'] = dataset.get_class_distribution()
            
            # 检查数据集质量
            self._check_dataset_quality(dataset, result)
            
            # 确定状态
            if result['issues']:
                result['status'] = 'has_issues'
            elif result['warnings']:
                result['status'] = 'has_warnings'
            else:
                result['status'] = 'healthy'
                
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f"加载失败: {str(e)}")
            logger.error(f"验证数据集 {dataset_name} 时出错: {e}")
        
        return result
    
    def _check_dataset_quality(self, dataset, result: Dict[str, Any]):
        """检查数据集质量"""
        # 检查样本数量
        if result['sample_count'] == 0:
            result['issues'].append("数据集为空")
        elif result['sample_count'] < 100:
            result['warnings'].append(f"样本数量较少: {result['sample_count']}")
        
        # 检查类别数量
        if result['class_count'] == 0:
            result['issues'].append("没有找到类别")
        elif result['class_count'] != 38 and dataset.config.expected_classes == 38:
            result['warnings'].append(f"类别数量不符合预期: {result['class_count']} vs {dataset.config.expected_classes}")
        
        # 检查类别平衡性
        if result['class_distribution']:
            counts = list(result['class_distribution'].values())
            if counts:
                min_count = min(counts)
                max_count = max(counts)
                if min_count > 0:
                    imbalance_ratio = max_count / min_count
                    if imbalance_ratio > 10:
                        result['warnings'].append(f"类别严重不平衡，比例: {imbalance_ratio:.1f}")
                    elif imbalance_ratio > 5:
                        result['warnings'].append(f"类别中度不平衡，比例: {imbalance_ratio:.1f}")
                
                # 检查样本过少的类别
                few_sample_classes = [cls for cls, count in result['class_distribution'].items() if count < 10]
                if few_sample_classes:
                    result['warnings'].append(f"{len(few_sample_classes)} 个类别样本过少 (<10个)")
    
    def _generate_summary(self, validations: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证总结"""
        summary = {
            'total_datasets': len(validations),
            'healthy_datasets': 0,
            'datasets_with_warnings': 0,
            'datasets_with_issues': 0,
            'failed_datasets': 0,
            'total_samples': 0,
            'total_classes': set()
        }
        
        for dataset_name, validation in validations.items():
            status = validation.get('status', 'unknown')
            
            if status == 'healthy':
                summary['healthy_datasets'] += 1
            elif status == 'has_warnings':
                summary['datasets_with_warnings'] += 1
            elif status == 'has_issues':
                summary['datasets_with_issues'] += 1
            else:
                summary['failed_datasets'] += 1
            
            summary['total_samples'] += validation.get('sample_count', 0)
            
            # 收集所有类别
            if 'class_distribution' in validation:
                summary['total_classes'].update(validation['class_distribution'].keys())
        
        summary['unique_classes'] = len(summary['total_classes'])
        summary['total_classes'] = list(summary['total_classes'])
        
        return summary
    
    def _generate_recommendations(self, validations: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 检查是否有失败的数据集
        failed_datasets = [name for name, val in validations.items() 
                          if val.get('status') in ['error', 'path_not_found']]
        if failed_datasets:
            recommendations.append(f"修复失败的数据集: {', '.join(failed_datasets)}")
        
        # 检查样本数量
        low_sample_datasets = [name for name, val in validations.items() 
                              if val.get('sample_count', 0) < 1000]
        if low_sample_datasets:
            recommendations.append(f"考虑增加样本数量较少的数据集: {', '.join(low_sample_datasets)}")
        
        # 检查类别不平衡
        imbalanced_datasets = []
        for name, val in validations.items():
            if any('不平衡' in warning for warning in val.get('warnings', [])):
                imbalanced_datasets.append(name)
        
        if imbalanced_datasets:
            recommendations.append(f"处理类别不平衡问题: {', '.join(imbalanced_datasets)}")
            recommendations.append("建议使用类别权重或数据增强来平衡训练")
        
        # 推荐最佳数据集
        healthy_datasets = [name for name, val in validations.items() 
                           if val.get('status') == 'healthy']
        if healthy_datasets:
            recommendations.append(f"推荐优先使用健康的数据集: {', '.join(healthy_datasets)}")
        
        # 如果没有问题
        if not recommendations:
            recommendations.append("所有数据集状态良好，可以开始训练")
        
        return recommendations
    
    def _determine_overall_status(self, validations: Dict[str, Any]) -> str:
        """确定总体状态"""
        statuses = [val.get('status', 'unknown') for val in validations.values()]
        
        if any(status in ['error', 'path_not_found'] for status in statuses):
            return 'critical_issues'
        elif any(status == 'has_issues' for status in statuses):
            return 'has_issues'
        elif any(status == 'has_warnings' for status in statuses):
            return 'has_warnings'
        elif all(status == 'healthy' for status in statuses):
            return 'healthy'
        else:
            return 'unknown'
    
    def print_validation_report(self, results: Dict[str, Any]):
        """打印验证报告"""
        print("📋 数据集验证报告")
        print("=" * 60)
        
        # 总体状态
        status_icons = {
            'healthy': '✅',
            'has_warnings': '⚠️',
            'has_issues': '🔧',
            'critical_issues': '❌',
            'unknown': '❓'
        }
        
        overall_status = results['overall_status']
        icon = status_icons.get(overall_status, '❓')
        print(f"\n{icon} 总体状态: {overall_status}")
        
        # 摘要
        summary = results['summary']
        print(f"\n📊 数据集摘要:")
        print(f"   总数据集: {summary['total_datasets']}")
        print(f"   健康: {summary['healthy_datasets']}")
        print(f"   有警告: {summary['datasets_with_warnings']}")
        print(f"   有问题: {summary['datasets_with_issues']}")
        print(f"   失败: {summary['failed_datasets']}")
        print(f"   总样本: {summary['total_samples']}")
        print(f"   唯一类别: {summary['unique_classes']}")
        
        # 各数据集详情
        print(f"\n📂 各数据集详情:")
        for dataset_name, validation in results['dataset_validations'].items():
            status = validation.get('status', 'unknown')
            icon = status_icons.get(status, '❓')
            
            print(f"\n   {icon} {dataset_name}:")
            print(f"      状态: {status}")
            print(f"      样本数: {validation.get('sample_count', 0)}")
            print(f"      类别数: {validation.get('class_count', 0)}")
            
            # 显示问题和警告
            issues = validation.get('issues', [])
            warnings = validation.get('warnings', [])
            
            if issues:
                print(f"      问题:")
                for issue in issues:
                    print(f"        - {issue}")
            
            if warnings:
                print(f"      警告:")
                for warning in warnings:
                    print(f"        - {warning}")
        
        # 建议
        recommendations = results['recommendations']
        print(f"\n💡 改进建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

def main():
    """主函数"""
    print("🔍 CropPilot 数据集验证工具")
    print("=" * 60)
    
    validator = DatasetValidator()
    
    # 执行验证
    print("开始验证数据集...")
    results = validator.validate_all_datasets()
    
    # 打印报告
    validator.print_validation_report(results)
    
    # 返回状态码
    overall_status = results['overall_status']
    if overall_status == 'healthy':
        print(f"\n🎉 验证完成: 所有数据集状态良好!")
        return 0
    elif overall_status in ['has_warnings', 'has_issues']:
        print(f"\n⚠️  验证完成: 发现一些问题，但可以继续")
        return 1
    else:
        print(f"\n❌ 验证失败: 存在严重问题")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)