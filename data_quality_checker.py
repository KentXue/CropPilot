#!/usr/bin/env python3
"""
数据质量检查工具
检查PlantVillage和百度AI Studio数据集的质量问题
"""

import os
import sys
from pathlib import Path
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import time
import hashlib

try:
    from PIL import Image, ImageStat
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL/numpy未安装，无法进行深度质量检查")
    print("   安装: pip install Pillow numpy")

class DataQualityChecker:
    def __init__(self):
        self.issues = {
            'corrupted_images': [],
            'duplicate_images': [],
            'low_quality_images': [],
            'size_anomalies': [],
            'format_issues': [],
            'label_issues': []
        }
        
    def check_image_corruption(self, image_path: str) -> bool:
        """检查图像是否损坏"""
        try:
            with Image.open(image_path) as img:
                img.verify()  # 验证图像完整性
            return True
        except Exception as e:
            self.issues['corrupted_images'].append({
                'path': image_path,
                'error': str(e)
            })
            return False
    
    def calculate_image_hash(self, image_path: str) -> str:
        """计算图像哈希值用于重复检测"""
        try:
            with Image.open(image_path) as img:
                # 转换为RGB并缩放到小尺寸计算哈希
                img = img.convert('RGB').resize((8, 8))
                img_array = np.array(img)
                return hashlib.md5(img_array.tobytes()).hexdigest()
        except:
            return None
    
    def check_image_quality(self, image_path: str) -> Dict[str, Any]:
        """检查图像质量"""
        try:
            with Image.open(image_path) as img:
                # 基本信息
                width, height = img.size
                mode = img.mode
                
                # 计算图像统计信息
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                stat = ImageStat.Stat(img)
                
                quality_info = {
                    'size': (width, height),
                    'mode': mode,
                    'mean_brightness': np.mean(stat.mean),
                    'std_brightness': np.mean(stat.stddev),
                    'file_size': os.path.getsize(image_path)
                }
                
                # 质量问题检测
                issues = []
                
                # 1. 尺寸过小
                if width < 100 or height < 100:
                    issues.append('too_small')
                
                # 2. 尺寸异常大
                if width > 5000 or height > 5000:
                    issues.append('too_large')
                
                # 3. 长宽比异常
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 5:
                    issues.append('extreme_aspect_ratio')
                
                # 4. 亮度异常
                if quality_info['mean_brightness'] < 20:
                    issues.append('too_dark')
                elif quality_info['mean_brightness'] > 235:
                    issues.append('too_bright')
                
                # 5. 对比度过低
                if quality_info['std_brightness'] < 10:
                    issues.append('low_contrast')
                
                # 6. 文件大小异常
                if quality_info['file_size'] < 1000:  # 小于1KB
                    issues.append('file_too_small')
                elif quality_info['file_size'] > 10 * 1024 * 1024:  # 大于10MB
                    issues.append('file_too_large')
                
                if issues:
                    self.issues['low_quality_images'].append({
                        'path': image_path,
                        'issues': issues,
                        'info': quality_info
                    })
                
                return quality_info
                
        except Exception as e:
            self.issues['format_issues'].append({
                'path': image_path,
                'error': str(e)
            })
            return None
    
    def check_dataset_balance(self, class_distribution: Dict[str, int]) -> Dict[str, Any]:
        """检查数据集类别平衡性"""
        if not class_distribution:
            return {'balanced': True, 'issues': []}
        
        counts = list(class_distribution.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = min(counts)
        max_count = max(counts)
        
        issues = []
        
        # 检查类别不平衡
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        if imbalance_ratio > 10:
            issues.append(f'severe_imbalance_ratio_{imbalance_ratio:.1f}')
        elif imbalance_ratio > 5:
            issues.append(f'moderate_imbalance_ratio_{imbalance_ratio:.1f}')
        
        # 检查样本过少的类别
        few_sample_classes = [cls for cls, count in class_distribution.items() if count < 100]
        if few_sample_classes:
            issues.append(f'few_samples_{len(few_sample_classes)}_classes')
        
        return {
            'balanced': len(issues) == 0,
            'issues': issues,
            'stats': {
                'mean': mean_count,
                'std': std_count,
                'min': min_count,
                'max': max_count,
                'imbalance_ratio': imbalance_ratio
            },
            'few_sample_classes': few_sample_classes
        }
    
    def check_plantvillage_dataset(self, base_path: str, max_samples: int = 1000) -> Dict[str, Any]:
        """检查PlantVillage数据集质量"""
        print("\n🔍 检查PlantVillage数据集质量...")
        
        color_path = os.path.join(base_path, "1.图像数据（病虫害识别核心）", 
                                 "plantvillage dataset", "color")
        
        if not os.path.exists(color_path):
            return {'status': 'not_found', 'message': f'路径不存在: {color_path}'}
        
        # 收集所有图像文件和类别信息
        image_files = []
        class_distribution = defaultdict(int)
        
        for root, dirs, files in os.walk(color_path):
            class_name = os.path.basename(root)
            if class_name != 'color':  # 跳过根目录
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(root, file)
                        image_files.append((image_path, class_name))
                        class_distribution[class_name] += 1
        
        print(f"   找到 {len(image_files)} 张图像，{len(class_distribution)} 个类别")
        
        # 采样检查（避免检查时间过长）
        if len(image_files) > max_samples:
            import random
            sampled_files = random.sample(image_files, max_samples)
            print(f"   采样 {max_samples} 张图像进行质量检查")
        else:
            sampled_files = image_files
        
        # 质量检查
        image_hashes = {}
        quality_stats = []
        
        for i, (image_path, class_name) in enumerate(sampled_files):
            if i % 100 == 0:
                print(f"   检查进度: {i}/{len(sampled_files)}")
            
            # 检查损坏
            if not self.check_image_corruption(image_path):
                continue
            
            # 检查重复
            img_hash = self.calculate_image_hash(image_path)
            if img_hash:
                if img_hash in image_hashes:
                    self.issues['duplicate_images'].append({
                        'original': image_hashes[img_hash],
                        'duplicate': image_path
                    })
                else:
                    image_hashes[img_hash] = image_path
            
            # 检查质量
            quality_info = self.check_image_quality(image_path)
            if quality_info:
                quality_stats.append(quality_info)
        
        # 检查类别平衡
        balance_info = self.check_dataset_balance(dict(class_distribution))
        
        return {
            'status': 'checked',
            'total_images': len(image_files),
            'sampled_images': len(sampled_files),
            'class_distribution': dict(class_distribution),
            'balance_info': balance_info,
            'quality_stats': {
                'mean_size': np.mean([q['size'][0] * q['size'][1] for q in quality_stats]) if quality_stats else 0,
                'mean_brightness': np.mean([q['mean_brightness'] for q in quality_stats]) if quality_stats else 0,
                'mean_file_size': np.mean([q['file_size'] for q in quality_stats]) if quality_stats else 0
            }
        }
    
    def check_baidu_dataset(self, base_path: str, max_samples: int = 500) -> Dict[str, Any]:
        """检查百度AI Studio数据集质量"""
        print("\n🔍 检查百度AI Studio数据集质量...")
        
        baidu_path = os.path.join(base_path, "1.图像数据（病虫害识别核心）", 
                                 "ai_challenger_pdr2018")
        
        if not os.path.exists(baidu_path):
            return {'status': 'not_found', 'message': f'路径不存在: {baidu_path}'}
        
        # 查找标注文件
        annotation_files = []
        for root, dirs, files in os.walk(baidu_path):
            for file in files:
                if file.endswith('.json'):
                    annotation_files.append(os.path.join(root, file))
        
        # 收集图像文件
        image_files = []
        for root, dirs, files in os.walk(baidu_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        print(f"   找到 {len(image_files)} 张图像，{len(annotation_files)} 个标注文件")
        
        # 采样检查
        if len(image_files) > max_samples:
            import random
            sampled_files = random.sample(image_files, max_samples)
            print(f"   采样 {max_samples} 张图像进行质量检查")
        else:
            sampled_files = image_files
        
        # 质量检查
        quality_stats = []
        for i, image_path in enumerate(sampled_files):
            if i % 50 == 0:
                print(f"   检查进度: {i}/{len(sampled_files)}")
            
            if not self.check_image_corruption(image_path):
                continue
            
            quality_info = self.check_image_quality(image_path)
            if quality_info:
                quality_stats.append(quality_info)
        
        return {
            'status': 'checked',
            'total_images': len(image_files),
            'sampled_images': len(sampled_files),
            'annotation_files': len(annotation_files),
            'quality_stats': {
                'mean_size': np.mean([q['size'][0] * q['size'][1] for q in quality_stats]) if quality_stats else 0,
                'mean_brightness': np.mean([q['mean_brightness'] for q in quality_stats]) if quality_stats else 0,
                'mean_file_size': np.mean([q['file_size'] for q in quality_stats]) if quality_stats else 0
            }
        }
    
    def generate_cleaning_recommendations(self) -> List[str]:
        """生成数据清洗建议"""
        recommendations = []
        
        # 损坏图像
        if self.issues['corrupted_images']:
            count = len(self.issues['corrupted_images'])
            recommendations.append(f"🔧 删除 {count} 张损坏的图像文件")
        
        # 重复图像
        if self.issues['duplicate_images']:
            count = len(self.issues['duplicate_images'])
            recommendations.append(f"🔄 删除 {count} 张重复图像以避免过拟合")
        
        # 低质量图像
        if self.issues['low_quality_images']:
            count = len(self.issues['low_quality_images'])
            quality_issues = defaultdict(int)
            for item in self.issues['low_quality_images']:
                for issue in item['issues']:
                    quality_issues[issue] += 1
            
            recommendations.append(f"⚠️  发现 {count} 张低质量图像:")
            for issue, issue_count in quality_issues.items():
                issue_desc = {
                    'too_small': '尺寸过小',
                    'too_large': '尺寸过大', 
                    'extreme_aspect_ratio': '长宽比异常',
                    'too_dark': '过暗',
                    'too_bright': '过亮',
                    'low_contrast': '对比度低',
                    'file_too_small': '文件过小',
                    'file_too_large': '文件过大'
                }.get(issue, issue)
                recommendations.append(f"   - {issue_desc}: {issue_count} 张")
        
        # 格式问题
        if self.issues['format_issues']:
            count = len(self.issues['format_issues'])
            recommendations.append(f"📁 修复 {count} 个图像格式问题")
        
        # 如果没有严重问题
        if not any(self.issues.values()):
            recommendations.append("✅ 数据集质量良好，无需大规模清洗")
            recommendations.append("💡 建议进行标准的预处理：尺寸标准化、数据增强等")
        
        return recommendations
    
    def print_quality_report(self, plantvillage_result: Dict, baidu_result: Dict):
        """打印质量检查报告"""
        print(f"\n📋 数据质量检查报告")
        print("=" * 60)
        
        # PlantVillage报告
        if plantvillage_result['status'] == 'checked':
            print(f"\n🌱 PlantVillage数据集:")
            print(f"   📊 总图像数: {plantvillage_result['total_images']}")
            print(f"   🔍 检查样本: {plantvillage_result['sampled_images']}")
            
            balance = plantvillage_result['balance_info']
            if balance['balanced']:
                print(f"   ⚖️  类别平衡: ✅ 良好")
            else:
                print(f"   ⚖️  类别平衡: ⚠️  存在问题")
                for issue in balance['issues']:
                    print(f"      - {issue}")
            
            stats = plantvillage_result['quality_stats']
            print(f"   📏 平均分辨率: {stats['mean_size']:.0f} 像素")
            print(f"   💡 平均亮度: {stats['mean_brightness']:.1f}")
            print(f"   📦 平均文件大小: {stats['mean_file_size']/1024:.1f} KB")
        
        # 百度数据集报告
        if baidu_result['status'] == 'checked':
            print(f"\n🇨🇳 百度AI Studio数据集:")
            print(f"   📊 总图像数: {baidu_result['total_images']}")
            print(f"   🔍 检查样本: {baidu_result['sampled_images']}")
            print(f"   📝 标注文件: {baidu_result['annotation_files']} 个")
            
            stats = baidu_result['quality_stats']
            print(f"   📏 平均分辨率: {stats['mean_size']:.0f} 像素")
            print(f"   💡 平均亮度: {stats['mean_brightness']:.1f}")
            print(f"   📦 平均文件大小: {stats['mean_file_size']/1024:.1f} KB")
        
        # 问题统计
        total_issues = sum(len(issues) for issues in self.issues.values())
        print(f"\n🚨 发现的问题:")
        print(f"   损坏图像: {len(self.issues['corrupted_images'])} 个")
        print(f"   重复图像: {len(self.issues['duplicate_images'])} 个") 
        print(f"   低质量图像: {len(self.issues['low_quality_images'])} 个")
        print(f"   格式问题: {len(self.issues['format_issues'])} 个")
        print(f"   总问题数: {total_issues} 个")

def main():
    """主函数"""
    print("🔍 CropPilot 数据质量检查工具")
    print("=" * 60)
    
    if not PIL_AVAILABLE:
        print("❌ 缺少必要依赖，请安装:")
        print("   pip install Pillow numpy")
        return
    
    base_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据"
    
    checker = DataQualityChecker()
    start_time = time.time()
    
    # 检查数据集质量
    plantvillage_result = checker.check_plantvillage_dataset(base_path)
    baidu_result = checker.check_baidu_dataset(base_path)
    
    # 打印报告
    checker.print_quality_report(plantvillage_result, baidu_result)
    
    # 生成建议
    recommendations = checker.generate_cleaning_recommendations()
    
    print(f"\n💡 数据清洗建议")
    print("=" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # 总结
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  质量检查完成，耗时: {elapsed_time:.1f} 秒")
    
    # 清洗必要性评估
    total_issues = sum(len(issues) for issues in checker.issues.values())
    if total_issues == 0:
        print(f"\n✅ 结论: 数据集质量优秀，可直接用于训练")
    elif total_issues < 100:
        print(f"\n⚠️  结论: 数据集质量良好，建议进行轻度清洗")
    else:
        print(f"\n🔧 结论: 发现较多问题，建议进行数据清洗")
    
    print(f"\n📋 下一步行动:")
    if total_issues > 0:
        print("1. 根据建议进行数据清洗")
        print("2. 重新运行质量检查验证清洗效果")
        print("3. 开始模型训练")
    else:
        print("1. 可以直接开始模型训练")
        print("2. 建议设置适当的数据增强策略")

if __name__ == "__main__":
    main()