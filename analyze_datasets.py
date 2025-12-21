#!/usr/bin/env python3
"""
深度分析数据集内容和结构
重新评估数据集的用途和关系
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_plantvillage_classes():
    """分析PlantVillage数据集的类别"""
    print("🔍 分析PlantVillage数据集类别...")
    
    color_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\1.图像数据（病虫害识别核心）\plantvillage dataset\color"
    
    if not os.path.exists(color_path):
        print(f"❌ 路径不存在: {color_path}")
        return {}
    
    # 获取所有类别目录
    classes = []
    class_counts = {}
    crop_types = defaultdict(list)
    
    for item in os.listdir(color_path):
        item_path = os.path.join(color_path, item)
        if os.path.isdir(item_path):
            classes.append(item)
            # 统计该类别的图片数量
            img_count = len([f for f in os.listdir(item_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[item] = img_count
            
            # 提取作物类型
            crop_name = item.split('___')[0] if '___' in item else item
            crop_types[crop_name].append(item)
    
    print(f"📊 PlantVillage数据集分析:")
    print(f"   总类别数: {len(classes)}")
    print(f"   作物种类: {len(crop_types)}")
    
    print(f"\n🌱 作物种类分布:")
    for crop, diseases in crop_types.items():
        total_images = sum(class_counts.get(disease, 0) for disease in diseases)
        print(f"   {crop}: {len(diseases)} 种病害, {total_images} 张图片")
        for disease in diseases[:3]:  # 只显示前3个病害
            print(f"     - {disease}: {class_counts.get(disease, 0)} 张")
        if len(diseases) > 3:
            print(f"     ... 还有 {len(diseases) - 3} 种病害")
    
    return crop_types, class_counts

def analyze_baidu_dataset():
    """分析百度AI Studio数据集"""
    print(f"\n🔍 分析百度AI Studio数据集...")
    
    baidu_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\1.图像数据（病虫害识别核心）\ai_challenger_pdr2018"
    
    if not os.path.exists(baidu_path):
        print(f"❌ 路径不存在: {baidu_path}")
        return {}
    
    # 查找所有文件
    all_files = []
    json_files = []
    image_files = []
    
    for root, dirs, files in os.walk(baidu_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
            
            if file.endswith('.json'):
                json_files.append(file_path)
            elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(file_path)
    
    print(f"📊 百度AI Studio数据集分析:")
    print(f"   总文件数: {len(all_files)}")
    print(f"   图像文件: {len(image_files)}")
    print(f"   JSON文件: {len(json_files)}")
    
    # 分析目录结构
    print(f"\n📁 目录结构:")
    for root, dirs, files in os.walk(baidu_path):
        level = root.replace(baidu_path, '').count(os.sep)
        indent = ' ' * 2 * level
        folder_name = os.path.basename(root)
        if folder_name:
            print(f"{indent}{folder_name}/ ({len(files)} 文件)")
    
    # 分析JSON标注文件
    if json_files:
        print(f"\n📝 标注文件分析:")
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_name = os.path.basename(json_file)
                print(f"   {file_name}:")
                
                if isinstance(data, list):
                    print(f"     类型: 列表, 长度: {len(data)}")
                    if data:
                        sample = data[0]
                        print(f"     样本字段: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                elif isinstance(data, dict):
                    print(f"     类型: 字典, 字段: {list(data.keys())}")
                
            except Exception as e:
                print(f"     ❌ 无法解析: {e}")
    
    return {
        'total_files': len(all_files),
        'image_files': len(image_files),
        'json_files': len(json_files),
        'json_paths': json_files
    }

def analyze_phenology_dataset():
    """分析物候数据集"""
    print(f"\n🔍 分析物候数据集...")
    
    phenology_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\2.生长数据（时间序列）\8313530"
    
    if not os.path.exists(phenology_path):
        print(f"❌ 路径不存在: {phenology_path}")
        return {}
    
    # 分析文件类型和数量
    file_types = defaultdict(int)
    total_size = 0
    
    for root, dirs, files in os.walk(phenology_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] += 1
            
            try:
                total_size += os.path.getsize(file_path)
            except:
                pass
    
    print(f"📊 物候数据集分析:")
    print(f"   总大小: {total_size / (1024**3):.1f} GB")
    print(f"   文件类型分布:")
    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"     {ext or '无扩展名'}: {count} 个")
    
    # 检查是否有说明文件
    readme_files = []
    for root, dirs, files in os.walk(phenology_path):
        for file in files:
            if any(keyword in file.lower() for keyword in ['readme', 'description', '说明', 'info']):
                readme_files.append(os.path.join(root, file))
    
    if readme_files:
        print(f"\n📖 说明文件:")
        for readme in readme_files:
            print(f"   {os.path.basename(readme)}")
    
    return {
        'total_size_gb': total_size / (1024**3),
        'file_types': dict(file_types),
        'readme_files': readme_files
    }

def generate_dataset_strategy():
    """生成数据集使用策略"""
    print(f"\n💡 数据集使用策略建议")
    print("=" * 60)
    
    print(f"🎯 推荐的训练策略:")
    print(f"1. **主要训练数据集**: PlantVillage Color")
    print(f"   - 优点: 54,000+张高质量标注图像，38个病害类别")
    print(f"   - 用途: 主要训练集，建立基础识别能力")
    print(f"   - 作物: 苹果、玉米、番茄、马铃薯等14种国际常见作物")
    
    print(f"\n2. **辅助训练数据集**: PlantVillage Segmented")
    print(f"   - 优点: 背景已分离，有助于提高识别精度")
    print(f"   - 用途: 精度优化阶段使用")
    print(f"   - 建议: 与Color数据集结合使用")
    
    print(f"\n3. **中国本土化数据集**: 百度AI Studio")
    print(f"   - 优点: 中国本土农作物，更符合实际应用场景")
    print(f"   - 用途: 模型微调和本土化适配")
    print(f"   - 注意: 需要先分析标注格式和类别映射")
    
    print(f"\n4. **物候数据集**: 时间序列数据")
    print(f"   - 用途: 提供时空上下文信息")
    print(f"   - 应用: 结合地理位置和时间信息增强识别准确性")
    print(f"   - 实现: 作为辅助特征，不是主要训练数据")
    
    print(f"\n🔄 建议的训练流程:")
    print(f"   阶段1: 使用PlantVillage Color训练基础模型")
    print(f"   阶段2: 使用PlantVillage Segmented进行精度优化")
    print(f"   阶段3: 使用百度数据集进行中国本土化微调")
    print(f"   阶段4: 集成物候数据提供上下文增强")

def main():
    """主函数"""
    print("🔬 CropPilot 数据集深度分析")
    print("=" * 60)
    
    # 分析各个数据集
    plantvillage_crops, plantvillage_counts = analyze_plantvillage_classes()
    baidu_info = analyze_baidu_dataset()
    phenology_info = analyze_phenology_dataset()
    
    # 生成策略建议
    generate_dataset_strategy()
    
    print(f"\n📋 分析完成!")
    print(f"建议: 重新设计数据集配置，明确各数据集的用途")

if __name__ == "__main__":
    main()