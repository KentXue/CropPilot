#!/usr/bin/env python3
"""
PlantVillage数据集加载器
实现PlantVillage数据集的专用加载器，包含英文到中文的类别映射
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from PIL import Image
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from torchvision import transforms
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    PIL_AVAILABLE = False

from src.dataset_config import get_dataset_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantVillageClassMapping:
    """PlantVillage类别映射管理"""
    
    def __init__(self):
        """初始化类别映射"""
        self.english_to_chinese = {
            # 苹果类
            'Apple___Apple_scab': '苹果_苹果黑星病',
            'Apple___Black_rot': '苹果_黑腐病',
            'Apple___Cedar_apple_rust': '苹果_雪松苹果锈病',
            'Apple___healthy': '苹果_健康',
            
            # 蓝莓类
            'Blueberry___healthy': '蓝莓_健康',
            
            # 樱桃类
            'Cherry_(including_sour)___Powdery_mildew': '樱桃_白粉病',
            'Cherry_(including_sour)___healthy': '樱桃_健康',
            
            # 玉米类
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': '玉米_灰斑病',
            'Corn_(maize)___Common_rust_': '玉米_普通锈病',
            'Corn_(maize)___Northern_Leaf_Blight': '玉米_北方叶枯病',
            'Corn_(maize)___healthy': '玉米_健康',
            
            # 葡萄类
            'Grape___Black_rot': '葡萄_黑腐病',
            'Grape___Esca_(Black_Measles)': '葡萄_黑麻疹病',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': '葡萄_叶枯病',
            'Grape___healthy': '葡萄_健康',
            
            # 橙子类
            'Orange___Haunglongbing_(Citrus_greening)': '橙子_黄龙病',
            
            # 桃子类
            'Peach___Bacterial_spot': '桃子_细菌性斑点病',
            'Peach___healthy': '桃子_健康',
            
            # 辣椒类
            'Pepper,_bell___Bacterial_spot': '甜椒_细菌性斑点病',
            'Pepper,_bell___healthy': '甜椒_健康',
            
            # 马铃薯类
            'Potato___Early_blight': '马铃薯_早疫病',
            'Potato___Late_blight': '马铃薯_晚疫病',
            'Potato___healthy': '马铃薯_健康',
            
            # 覆盆子类
            'Raspberry___healthy': '覆盆子_健康',
            
            # 大豆类
            'Soybean___healthy': '大豆_健康',
            
            # 南瓜类
            'Squash___Powdery_mildew': '南瓜_白粉病',
            
            # 草莓类
            'Strawberry___Leaf_scorch': '草莓_叶焦病',
            'Strawberry___healthy': '草莓_健康',
            
            # 番茄类
            'Tomato___Bacterial_spot': '番茄_细菌性斑点病',
            'Tomato___Early_blight': '番茄_早疫病',
            'Tomato___Late_blight': '番茄_晚疫病',
            'Tomato___Leaf_Mold': '番茄_叶霉病',
            'Tomato___Septoria_leaf_spot': '番茄_斑点病',
            'Tomato___Spider_mites Two-spotted_spider_mite': '番茄_红蜘蛛',
            'Tomato___Target_Spot': '番茄_靶斑病',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': '番茄_黄化曲叶病毒',
            'Tomato___Tomato_mosaic_virus': '番茄_花叶病毒',
            'Tomato___healthy': '番茄_健康'
        }
        
        # 创建反向映射
        self.chinese_to_english = {v: k for k, v in self.english_to_chinese.items()}
        
        # 作物分类
        self.crop_categories = {
            '果树类': ['苹果', '蓝莓', '樱桃', '葡萄', '橙子', '桃子', '覆盆子', '草莓'],
            '粮食作物': ['玉米', '大豆', '马铃薯'],
            '蔬菜类': ['甜椒', '南瓜', '番茄']
        }
        
        # 病害类型分类
        self.disease_categories = {
            '真菌病害': ['黑星病', '黑腐病', '锈病', '白粉病', '叶枯病', '早疫病', '晚疫病', '叶霉病', '斑点病', '靶斑病'],
            '细菌病害': ['细菌性斑点病'],
            '病毒病害': ['黄化曲叶病毒', '花叶病毒'],
            '虫害': ['红蜘蛛'],
            '生理病害': ['叶焦病', '黄龙病'],
            '健康': ['健康']
        }
    
    def get_chinese_name(self, english_name: str) -> str:
        """获取中文名称"""
        return self.english_to_chinese.get(english_name, english_name)
    
    def get_english_name(self, chinese_name: str) -> str:
        """获取英文名称"""
        return self.chinese_to_english.get(chinese_name, chinese_name)
    
    def get_crop_name(self, class_name: str) -> str:
        """从类别名称中提取作物名称"""
        if '_' in class_name:
            return class_name.split('_')[0]
        return class_name.split('___')[0] if '___' in class_name else class_name
    
    def get_disease_name(self, class_name: str) -> str:
        """从类别名称中提取病害名称"""
        if '_' in class_name:
            return class_name.split('_')[1]
        return class_name.split('___')[1] if '___' in class_name else 'unknown'
    
    def get_crop_category(self, crop_name: str) -> str:
        """获取作物类别"""
        for category, crops in self.crop_categories.items():
            if crop_name in crops:
                return category
        return '其他'
    
    def get_disease_category(self, disease_name: str) -> str:
        """获取病害类别"""
        for category, diseases in self.disease_categories.items():
            if any(disease in disease_name for disease in diseases):
                return category
        return '其他'
    
    def get_all_classes(self) -> List[str]:
        """获取所有类别（英文）"""
        return list(self.english_to_chinese.keys())
    
    def get_all_chinese_classes(self) -> List[str]:
        """获取所有类别（中文）"""
        return list(self.english_to_chinese.values())
    
    def get_class_statistics(self) -> Dict[str, Any]:
        """获取类别统计信息"""
        crop_count = defaultdict(int)
        disease_count = defaultdict(int)
        category_count = defaultdict(int)
        
        for english_name, chinese_name in self.english_to_chinese.items():
            crop = self.get_crop_name(chinese_name)
            disease = self.get_disease_name(chinese_name)
            category = self.get_crop_category(crop)
            
            crop_count[crop] += 1
            disease_count[disease] += 1
            category_count[category] += 1
        
        return {
            'total_classes': len(self.english_to_chinese),
            'crop_distribution': dict(crop_count),
            'disease_distribution': dict(disease_count),
            'category_distribution': dict(category_count),
            'crop_categories': self.crop_categories,
            'disease_categories': self.disease_categories
        }

class PlantVillageDatasetLoader:
    """PlantVillage数据集专用加载器"""
    
    def __init__(self, dataset_type: str = 'color'):
        """
        初始化加载器
        
        Args:
            dataset_type: 数据集类型 ('color', 'grayscale', 'segmented')
        """
        self.dataset_type = dataset_type
        self.config = get_dataset_config()
        self.class_mapping = PlantVillageClassMapping()
        
        # 获取数据集配置
        if dataset_type not in self.config.plantvillage_datasets:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        self.dataset_config = self.config.plantvillage_datasets[dataset_type]
        
        # 数据集信息
        self.dataset_info = None
        self.class_distribution = None
        
        logger.info(f"PlantVillage {dataset_type} 数据集加载器初始化完成")
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """分析数据集结构和内容"""
        if not os.path.exists(self.dataset_config.path):
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_config.path}")
        
        logger.info(f"开始分析 {self.dataset_type} 数据集...")
        
        # 获取所有类别目录
        class_dirs = [d for d in os.listdir(self.dataset_config.path) 
                     if os.path.isdir(os.path.join(self.dataset_config.path, d))]
        class_dirs.sort()
        
        # 统计每个类别的图像数量
        class_stats = {}
        total_images = 0
        
        for class_name in class_dirs:
            class_path = os.path.join(self.dataset_config.path, class_name)
            
            # 统计图像文件
            image_count = 0
            image_formats = defaultdict(int)
            
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.dataset_config.image_extensions:
                        image_count += 1
                        image_formats[ext] += 1
            
            # 获取中文名称
            chinese_name = self.class_mapping.get_chinese_name(class_name)
            crop_name = self.class_mapping.get_crop_name(chinese_name)
            disease_name = self.class_mapping.get_disease_name(chinese_name)
            
            class_stats[class_name] = {
                'chinese_name': chinese_name,
                'crop': crop_name,
                'disease': disease_name,
                'image_count': image_count,
                'image_formats': dict(image_formats)
            }
            
            total_images += image_count
        
        # 创建数据集信息
        self.dataset_info = {
            'dataset_type': self.dataset_type,
            'dataset_path': self.dataset_config.path,
            'total_classes': len(class_dirs),
            'total_images': total_images,
            'class_statistics': class_stats,
            'class_mapping_stats': self.class_mapping.get_class_statistics(),
            'input_size': self.dataset_config.input_size
        }
        
        # 保存类别分布
        self.class_distribution = {k: v['image_count'] for k, v in class_stats.items()}
        
        logger.info(f"数据集分析完成: {len(class_dirs)} 个类别, {total_images} 张图像")
        
        return self.dataset_info
    
    def create_train_val_split(self, 
                              train_ratio: float = 0.8,
                              stratified: bool = True,
                              random_seed: int = 42) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        创建训练/验证集分割
        
        Args:
            train_ratio: 训练集比例
            stratified: 是否分层采样
            random_seed: 随机种子
            
        Returns:
            (train_split, val_split) - 每个都是 {class_name: [image_paths]} 的字典
        """
        if self.dataset_info is None:
            self.analyze_dataset()
        
        random.seed(random_seed)
        
        train_split = {}
        val_split = {}
        
        for class_name, class_info in self.dataset_info['class_statistics'].items():
            class_path = os.path.join(self.dataset_config.path, class_name)
            
            # 获取所有图像文件
            image_files = []
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.dataset_config.image_extensions:
                        image_files.append(file_path)
            
            # 随机打乱
            random.shuffle(image_files)
            
            # 分割
            split_idx = int(len(image_files) * train_ratio)
            train_split[class_name] = image_files[:split_idx]
            val_split[class_name] = image_files[split_idx:]
        
        # 统计分割结果
        train_total = sum(len(files) for files in train_split.values())
        val_total = sum(len(files) for files in val_split.values())
        
        logger.info(f"数据集分割完成: 训练集 {train_total} 张, 验证集 {val_total} 张")
        
        return train_split, val_split
    
    def generate_dataset_report(self, output_path: Optional[str] = None) -> str:
        """
        生成数据集分析报告
        
        Args:
            output_path: 输出文件路径，如果为None则返回字符串
            
        Returns:
            报告内容
        """
        if self.dataset_info is None:
            self.analyze_dataset()
        
        # 生成报告内容
        report_lines = [
            f"# PlantVillage {self.dataset_type.upper()} 数据集分析报告",
            f"",
            f"## 基本信息",
            f"- **数据集类型**: {self.dataset_type}",
            f"- **数据集路径**: {self.dataset_info['dataset_path']}",
            f"- **总类别数**: {self.dataset_info['total_classes']}",
            f"- **总图像数**: {self.dataset_info['total_images']}",
            f"- **输入尺寸**: {self.dataset_info['input_size']}",
            f"",
            f"## 类别映射统计",
        ]
        
        mapping_stats = self.dataset_info['class_mapping_stats']
        report_lines.extend([
            f"- **总类别数**: {mapping_stats['total_classes']}",
            f"- **作物类别分布**: {mapping_stats['category_distribution']}",
            f"- **主要作物**: {', '.join(list(mapping_stats['crop_distribution'].keys())[:10])}",
            f"",
            f"## 类别详细信息",
            f""
        ])
        
        # 按图像数量排序显示类别
        sorted_classes = sorted(
            self.dataset_info['class_statistics'].items(),
            key=lambda x: x[1]['image_count'],
            reverse=True
        )
        
        report_lines.append("| 英文名称 | 中文名称 | 作物 | 病害 | 图像数量 |")
        report_lines.append("|----------|----------|------|------|----------|")
        
        for class_name, class_info in sorted_classes:
            report_lines.append(
                f"| {class_name} | {class_info['chinese_name']} | "
                f"{class_info['crop']} | {class_info['disease']} | {class_info['image_count']} |"
            )
        
        # 添加分布统计
        report_lines.extend([
            f"",
            f"## 数据分布分析",
            f"",
            f"### 作物分布",
        ])
        
        for crop, count in mapping_stats['crop_distribution'].items():
            report_lines.append(f"- **{crop}**: {count} 个类别")
        
        report_lines.extend([
            f"",
            f"### 病害类型分布",
        ])
        
        for disease_type, diseases in mapping_stats['disease_categories'].items():
            count = sum(1 for class_info in self.dataset_info['class_statistics'].values()
                       if any(disease in class_info['disease'] for disease in diseases))
            report_lines.append(f"- **{disease_type}**: {count} 个类别")
        
        report_content = "\n".join(report_lines)
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"数据集报告已保存到: {output_path}")
        
        return report_content
    
    def get_default_transforms(self, is_training: bool = True):
        """获取默认的图像变换"""
        if not PIL_AVAILABLE:
            return None
        
        if is_training:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

# 便捷函数
def create_plantvillage_loader(dataset_type: str = 'color') -> PlantVillageDatasetLoader:
    """创建PlantVillage数据集加载器"""
    return PlantVillageDatasetLoader(dataset_type)

def get_plantvillage_class_mapping() -> PlantVillageClassMapping:
    """获取PlantVillage类别映射"""
    return PlantVillageClassMapping()

if __name__ == "__main__":
    # 测试PlantVillage加载器
    print("🧪 PlantVillage数据集加载器测试")
    print("=" * 60)
    
    if not PIL_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试类别映射
        print("📋 测试类别映射...")
        mapping = get_plantvillage_class_mapping()
        stats = mapping.get_class_statistics()
        
        print(f"✅ 类别映射统计:")
        print(f"   总类别数: {stats['total_classes']}")
        print(f"   作物类别: {stats['category_distribution']}")
        print(f"   前5个类别映射:")
        for i, (eng, chn) in enumerate(list(mapping.english_to_chinese.items())[:5]):
            print(f"     {eng} -> {chn}")
        
        # 测试数据集加载器
        print(f"\n🔍 测试color数据集加载器...")
        loader = create_plantvillage_loader('color')
        
        # 分析数据集
        dataset_info = loader.analyze_dataset()
        print(f"✅ 数据集分析完成:")
        print(f"   数据集类型: {dataset_info['dataset_type']}")
        print(f"   总类别数: {dataset_info['total_classes']}")
        print(f"   总图像数: {dataset_info['total_images']}")
        
        # 测试训练/验证集分割
        print(f"\n📊 测试数据集分割...")
        train_split, val_split = loader.create_train_val_split(train_ratio=0.8)
        
        train_total = sum(len(files) for files in train_split.values())
        val_total = sum(len(files) for files in val_split.values())
        
        print(f"✅ 数据集分割完成:")
        print(f"   训练集: {train_total} 张图像")
        print(f"   验证集: {val_total} 张图像")
        print(f"   分割比例: {train_total/(train_total+val_total):.2f}")
        
        # 生成报告
        print(f"\n📄 生成数据集报告...")
        report_path = f"PlantVillage_{loader.dataset_type}_dataset_report.md"
        report = loader.generate_dataset_report(report_path)
        print(f"✅ 报告已生成: {report_path}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ PlantVillage数据集加载器测试完成")