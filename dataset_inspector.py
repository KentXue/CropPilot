#!/usr/bin/env python3
"""
数据集检查工具
用于分析CropPilot AI图像识别项目的数据集结构和内容
"""

import os
import sys
from pathlib import Path
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import time

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL未安装，无法检查图像详情")
    print("   安装: pip install Pillow")

def check_path_exists(path: str) -> bool:
    """检查路径是否存在"""
    return os.path.exists(path) and os.path.isdir(path)

def get_directory_size(path: str) -> Tuple[int, str]:
    """获取目录大小"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"计算大小时出错: {e}")
        return 0, "未知"
    
    # 转换为可读格式
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return total_size, f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return total_size, f"{total_size:.1f} TB"

def count_files_by_extension(path: str) -> Dict[str, int]:
    """统计不同扩展名的文件数量"""
    extensions = defaultdict(int)
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                extensions[ext] += 1
    except Exception as e:
        print(f"统计文件时出错: {e}")
    return dict(extensions)

def analyze_image_dataset(path: str, max_samples: int = 100) -> Dict[str, Any]:
    """分析图像数据集"""
    analysis = {
        'total_images': 0,
        'image_formats': Counter(),
        'image_sizes': [],
        'directory_structure': {},
        'sample_files': []
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    try:
        for root, dirs, files in os.walk(path):
            # 记录目录结构
            rel_path = os.path.relpath(root, path)
            if rel_path != '.':
                analysis['directory_structure'][rel_path] = len(files)
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    analysis['total_images'] += 1
                    analysis['image_formats'][ext] += 1
                    
                    # 采样分析图像详情
                    if len(analysis['sample_files']) < max_samples and PIL_AVAILABLE:
                        try:
                            img_path = os.path.join(root, file)
                            with Image.open(img_path) as img:
                                analysis['image_sizes'].append(img.size)
                                analysis['sample_files'].append({
                                    'path': os.path.relpath(img_path, path),
                                    'size': img.size,
                                    'mode': img.mode,
                                    'format': img.format
                                })
                        except Exception as e:
                            print(f"无法读取图像 {file}: {e}")
    
    except Exception as e:
        print(f"分析图像数据集时出错: {e}")
    
    return analysis

def inspect_plantvillage_dataset(base_path: str) -> Dict[str, Any]:
    """检查PlantVillage数据集"""
    print("\n🔍 检查PlantVillage数据集...")
    
    # 检查三个PlantVillage子数据集
    plantvillage_base = os.path.join(base_path, "1.图像数据（病虫害识别核心）", "plantvillage dataset")
    
    datasets = {
        'color': os.path.join(plantvillage_base, "color"),
        'grayscale': os.path.join(plantvillage_base, "grayscale"), 
        'segmented': os.path.join(plantvillage_base, "segmented")
    }
    
    results = {}
    total_images = 0
    all_classes = {}
    
    for dataset_type, dataset_path in datasets.items():
        if not check_path_exists(dataset_path):
            results[dataset_type] = {
                'status': 'not_found',
                'message': f'路径不存在: {dataset_path}'
            }
            continue
        
        # 获取基本信息
        size_bytes, size_str = get_directory_size(dataset_path)
        file_types = count_files_by_extension(dataset_path)
        
        # 分析图像数据
        image_analysis = analyze_image_dataset(dataset_path)
        
        # 查找类别信息
        class_info = {}
        for root, dirs, files in os.walk(dataset_path):
            if dirs:  # 如果有子目录，可能是类别目录
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    img_count = len([f for f in os.listdir(dir_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    if img_count > 0:
                        class_info[dir_name] = img_count
                        all_classes[dir_name] = all_classes.get(dir_name, 0) + img_count
            break  # 只检查第一层目录
        
        total_images += image_analysis['total_images']
        
        results[dataset_type] = {
            'status': 'found',
            'path': dataset_path,
            'size': size_str,
            'file_types': file_types,
            'total_images': image_analysis['total_images'],
            'image_formats': dict(image_analysis['image_formats']),
            'directory_structure': image_analysis['directory_structure'],
            'classes': class_info,
            'sample_images': image_analysis['sample_files'][:3]
        }
    
    return {
        'status': 'found' if any(r.get('status') == 'found' for r in results.values()) else 'not_found',
        'datasets': results,
        'total_images': total_images,
        'total_classes': len(all_classes),
        'class_distribution': all_classes
    }

def inspect_baidu_dataset(base_path: str) -> Dict[str, Any]:
    """检查百度AI Studio数据集"""
    print("\n🔍 检查百度AI Studio数据集...")
    
    baidu_path = os.path.join(base_path, "1.图像数据（病虫害识别核心）", "ai_challenger_pdr2018")
    
    if not check_path_exists(baidu_path):
        return {
            'status': 'not_found',
            'message': f'路径不存在: {baidu_path}'
        }
    
    # 获取基本信息
    size_bytes, size_str = get_directory_size(baidu_path)
    file_types = count_files_by_extension(baidu_path)
    
    # 分析图像数据
    image_analysis = analyze_image_dataset(baidu_path)
    
    # 查找标注文件
    annotation_files = []
    for root, dirs, files in os.walk(baidu_path):
        for file in files:
            if file.endswith(('.json', '.csv', '.txt', '.xml')):
                annotation_files.append(os.path.join(root, file))
    
    return {
        'status': 'found',
        'path': baidu_path,
        'size': size_str,
        'file_types': file_types,
        'total_images': image_analysis['total_images'],
        'image_formats': dict(image_analysis['image_formats']),
        'directory_structure': image_analysis['directory_structure'],
        'annotation_files': annotation_files,
        'sample_images': image_analysis['sample_files'][:5]
    }

def inspect_phenology_dataset(base_path: str) -> Dict[str, Any]:
    """检查物候数据集"""
    print("\n🔍 检查ChinaCropPhen1km物候数据集...")
    
    phenology_path = os.path.join(base_path, "2.生长数据（时间序列）", "8313530")
    
    if not check_path_exists(phenology_path):
        return {
            'status': 'not_found',
            'message': f'路径不存在: {phenology_path}'
        }
    
    # 获取基本信息
    size_bytes, size_str = get_directory_size(phenology_path)
    file_types = count_files_by_extension(phenology_path)
    
    # 查找数据文件
    data_files = []
    for root, dirs, files in os.walk(phenology_path):
        for file in files:
            if file.endswith(('.tif', '.tiff', '.nc', '.hdf', '.dat')):
                data_files.append({
                    'name': file,
                    'path': os.path.relpath(os.path.join(root, file), phenology_path),
                    'size': os.path.getsize(os.path.join(root, file))
                })
    
    return {
        'status': 'found',
        'path': phenology_path,
        'size': size_str,
        'file_types': file_types,
        'data_files': data_files[:10],  # 只显示前10个文件
        'total_data_files': len(data_files)
    }

def print_dataset_summary(dataset_name: str, info: Dict[str, Any]):
    """打印数据集摘要"""
    print(f"\n📊 {dataset_name} 数据集分析结果")
    print("=" * 60)
    
    if info['status'] == 'not_found':
        print(f"❌ {info['message']}")
        return
    
    # 处理PlantVillage的特殊结构
    if 'datasets' in info:
        print(f"✅ PlantVillage数据集包含 {len(info['datasets'])} 个子数据集")
        print(f"🖼️  总图像数: {info['total_images']}")
        print(f"🏷️  总类别数: {info['total_classes']}")
        
        for dataset_type, dataset_info in info['datasets'].items():
            if dataset_info['status'] == 'found':
                print(f"\n📂 {dataset_type.upper()} 数据集:")
                print(f"   📦 大小: {dataset_info['size']}")
                print(f"   🖼️  图像数: {dataset_info['total_images']}")
                print(f"   🏷️  类别数: {len(dataset_info['classes'])}")
                
                if dataset_info['image_formats']:
                    print("   📋 图像格式:")
                    for fmt, count in dataset_info['image_formats'].items():
                        print(f"      {fmt}: {count} 张")
            else:
                print(f"\n❌ {dataset_type.upper()}: {dataset_info['message']}")
        
        if info['class_distribution']:
            print(f"\n🏷️  类别分布 (前10个):")
            sorted_classes = sorted(info['class_distribution'].items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_classes[:10]:
                print(f"   {class_name}: {count} 张")
            if len(sorted_classes) > 10:
                print(f"   ... 还有 {len(sorted_classes) - 10} 个类别")
        return
    
    print(f"✅ 路径: {info['path']}")
    print(f"📦 大小: {info['size']}")
    
    if 'total_images' in info:
        print(f"🖼️  图像总数: {info['total_images']}")
        if info['image_formats']:
            print("📋 图像格式:")
            for fmt, count in info['image_formats'].items():
                print(f"   {fmt}: {count} 张")
    
    if 'classes' in info and info['classes']:
        print(f"🏷️  类别数量: {len(info['classes'])}")
        print("📂 类别分布 (前10个):")
        sorted_classes = sorted(info['classes'].items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:10]:
            print(f"   {class_name}: {count} 张")
        if len(sorted_classes) > 10:
            print(f"   ... 还有 {len(sorted_classes) - 10} 个类别")
    
    if 'annotation_files' in info and info['annotation_files']:
        print(f"📝 标注文件: {len(info['annotation_files'])} 个")
        for ann_file in info['annotation_files'][:3]:
            print(f"   {os.path.basename(ann_file)}")
    
    if 'total_data_files' in info:
        print(f"📊 数据文件: {info['total_data_files']} 个")
    
    if 'file_types' in info:
        print("📁 文件类型分布:")
        for ext, count in sorted(info['file_types'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   {ext or '无扩展名'}: {count} 个")

def generate_recommendations(plantvillage_info: Dict, baidu_info: Dict, phenology_info: Dict) -> List[str]:
    """生成数据集使用建议"""
    recommendations = []
    
    # 检查数据集可用性
    available_datasets = []
    if plantvillage_info['status'] == 'found':
        available_datasets.append('PlantVillage')
    if baidu_info['status'] == 'found':
        available_datasets.append('百度AI Studio')
    if phenology_info['status'] == 'found':
        available_datasets.append('物候数据')
    
    if len(available_datasets) == 3:
        recommendations.append("✅ 所有三个数据集都可用，可以实现完整的AI识别系统")
    elif len(available_datasets) >= 1:
        recommendations.append(f"⚠️  只有 {len(available_datasets)} 个数据集可用: {', '.join(available_datasets)}")
    else:
        recommendations.append("❌ 没有找到可用的数据集，请检查路径配置")
        return recommendations
    
    # PlantVillage数据集建议
    if plantvillage_info['status'] == 'found':
        total_images = plantvillage_info.get('total_images', 0)
        if total_images > 40000:
            recommendations.append("🎯 PlantVillage数据集图像充足，适合作为主要训练数据")
            
            # 检查哪个子数据集最适合
            if 'datasets' in plantvillage_info:
                color_images = plantvillage_info['datasets'].get('color', {}).get('total_images', 0)
                if color_images > 30000:
                    recommendations.append("🌈 建议优先使用color数据集进行训练（彩色图像效果更好）")
                
                segmented_images = plantvillage_info['datasets'].get('segmented', {}).get('total_images', 0)
                if segmented_images > 20000:
                    recommendations.append("✂️  segmented数据集可用于提高识别精度（背景已分离）")
                    
        elif total_images > 10000:
            recommendations.append("⚠️  PlantVillage数据集图像较少，建议结合百度数据集")
        else:
            recommendations.append("❌ PlantVillage数据集图像过少，可能影响训练效果")
    
    # 百度数据集建议
    if baidu_info['status'] == 'found':
        if baidu_info.get('annotation_files'):
            recommendations.append("📝 百度数据集包含标注文件，可用于验证和微调")
        else:
            recommendations.append("⚠️  百度数据集缺少标注文件，需要进一步检查")
    
    # 物候数据建议
    if phenology_info['status'] == 'found':
        recommendations.append("🌱 物候数据可用于提供上下文信息，增强识别准确性")
    
    # 实施建议
    if len(available_datasets) >= 2:
        recommendations.append("🚀 建议按计划实施：先用PlantVillage训练，再用百度数据验证")
        recommendations.append("💡 可以实现预期的85%+识别准确率")
    
    return recommendations

def main():
    """主函数"""
    print("🌾 CropPilot AI图像识别数据集检查工具")
    print("=" * 60)
    
    # 数据集基础路径
    base_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据"
    
    print(f"📁 数据集基础路径: {base_path}")
    
    if not check_path_exists(base_path):
        print(f"❌ 基础路径不存在: {base_path}")
        print("\n💡 请确认数据集路径是否正确")
        return
    
    print("✅ 基础路径存在，开始检查各个数据集...")
    
    # 检查各个数据集
    start_time = time.time()
    
    plantvillage_info = inspect_plantvillage_dataset(base_path)
    baidu_info = inspect_baidu_dataset(base_path)
    phenology_info = inspect_phenology_dataset(base_path)
    
    # 打印结果
    print_dataset_summary("PlantVillage", plantvillage_info)
    print_dataset_summary("百度AI Studio", baidu_info)
    print_dataset_summary("物候数据", phenology_info)
    
    # 生成建议
    recommendations = generate_recommendations(plantvillage_info, baidu_info, phenology_info)
    
    print(f"\n💡 数据集使用建议")
    print("=" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    # 总结
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  检查完成，耗时: {elapsed_time:.1f} 秒")
    
    print(f"\n📋 下一步行动:")
    print("1. 如果数据集都可用，可以开始执行任务计划")
    print("2. 如果有问题，请先解决数据集路径或格式问题")
    print("3. 建议先运行一个小规模测试来验证数据加载")

if __name__ == "__main__":
    main()