#!/usr/bin/env python3
"""
数据集管理器
统一管理PlantVillage、百度AI Studio和物候数据集
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from PIL import Image
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    PIL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    PIL_AVAILABLE = False

from src.dataset_config import get_dataset_config, ImageDatasetConfig, PhenologyDatasetConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseDataset(Dataset):
    """植物病害数据集类"""
    
    def __init__(self, 
                 dataset_config: ImageDatasetConfig,
                 transform=None,
                 max_samples_per_class: Optional[int] = None,
                 class_subset: Optional[List[str]] = None):
        """
        初始化数据集
        
        Args:
            dataset_config: 数据集配置
            transform: 图像变换
            max_samples_per_class: 每个类别最大样本数（用于快速测试）
            class_subset: 类别子集（只加载指定类别）
        """
        self.config = dataset_config
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.class_subset = class_subset
        
        # 存储数据
        self.samples = []  # (image_path, class_index)
        self.classes = []  # 类别名称列表
        self.class_to_idx = {}  # 类别名称到索引的映射
        
        # 加载数据
        self._load_dataset()
        
        logger.info(f"数据集 {self.config.name} 加载完成: {len(self.samples)} 个样本, {len(self.classes)} 个类别")
    
    def _load_dataset(self):
        """加载数据集"""
        if not os.path.exists(self.config.path):
            raise FileNotFoundError(f"数据集路径不存在: {self.config.path}")
        
        if self.config.has_subdirectories:
            self._load_from_subdirectories()
        else:
            self._load_from_annotations()
    
    def _load_from_subdirectories(self):
        """从子目录结构加载数据（PlantVillage格式）"""
        class_dirs = [d for d in os.listdir(self.config.path) 
                     if os.path.isdir(os.path.join(self.config.path, d))]
        
        # 过滤类别子集
        if self.class_subset:
            class_dirs = [d for d in class_dirs if d in self.class_subset]
        
        class_dirs.sort()
        self.classes = class_dirs
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 收集样本
        for class_name in self.classes:
            class_path = os.path.join(self.config.path, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取该类别的所有图像文件
            image_files = []
            for ext in self.config.image_extensions:
                pattern = os.path.join(class_path, f"*{ext}")
                import glob
                image_files.extend(glob.glob(pattern))
            
            # 限制每个类别的样本数
            if self.max_samples_per_class and len(image_files) > self.max_samples_per_class:
                image_files = random.sample(image_files, self.max_samples_per_class)
            
            # 添加样本
            for img_path in image_files:
                self.samples.append((img_path, class_idx))
    
    def _load_from_annotations(self):
        """从标注文件加载数据（百度AI Studio格式）"""
        # 查找JSON标注文件
        annotation_files = []
        for root, dirs, files in os.walk(self.config.path):
            for file in files:
                if file.endswith('.json'):
                    annotation_files.append(os.path.join(root, file))
        
        if not annotation_files:
            logger.warning(f"在 {self.config.path} 中未找到JSON标注文件，尝试直接加载图像")
            self._load_images_directly()
            return
        
        # 加载标注数据
        all_annotations = []
        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_annotations.extend(data)
                    elif isinstance(data, dict):
                        all_annotations.append(data)
            except Exception as e:
                logger.warning(f"无法加载标注文件 {ann_file}: {e}")
        
        if not all_annotations:
            logger.warning("未找到有效的标注数据，尝试直接加载图像")
            self._load_images_directly()
            return
        
        # 处理标注数据
        self._process_annotations(all_annotations)
    
    def _load_images_directly(self):
        """直接加载图像文件（无标注）"""
        image_files = []
        for root, dirs, files in os.walk(self.config.path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.config.image_extensions):
                    image_files.append(os.path.join(root, file))
        
        # 创建单一类别
        self.classes = ['unknown']
        self.class_to_idx = {'unknown': 0}
        
        # 添加样本
        for img_path in image_files:
            self.samples.append((img_path, 0))
    
    def _process_annotations(self, annotations: List[Dict]):
        """处理标注数据"""
        # 提取所有类别
        all_classes = set()
        valid_samples = []
        
        for ann in annotations:
            # 尝试不同的标注格式
            image_path = None
            label = None
            
            # 格式1: {"image": "path", "label": "class"}
            if 'image' in ann and 'label' in ann:
                image_path = ann['image']
                label = ann['label']
            # 格式2: {"filename": "path", "class": "label"}
            elif 'filename' in ann and 'class' in ann:
                image_path = ann['filename']
                label = ann['class']
            # 格式3: {"image_path": "path", "disease": "label"}
            elif 'image_path' in ann and 'disease' in ann:
                image_path = ann['image_path']
                label = ann['disease']
            
            if image_path and label:
                # 构建完整路径
                full_path = os.path.join(self.config.path, image_path)
                if os.path.exists(full_path):
                    all_classes.add(label)
                    valid_samples.append((full_path, label))
        
        # 设置类别映射
        self.classes = sorted(list(all_classes))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 转换样本
        for img_path, label in valid_samples:
            class_idx = self.class_to_idx[label]
            self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        img_path, class_idx = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"无法加载图像 {img_path}: {e}")
            # 返回黑色图像作为备用
            image = Image.new('RGB', self.config.input_size, (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = Counter()
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            distribution[class_name] += 1
        return dict(distribution)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            'name': self.config.name,
            'path': self.config.path,
            'total_samples': len(self.samples),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': self.get_class_distribution(),
            'input_size': self.config.input_size
        }

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self):
        """初始化数据集管理器"""
        self.config = get_dataset_config()
        self.datasets = {}
        self.dataset_info = {}
        
        logger.info("数据集管理器初始化完成")
    
    def load_dataset(self, 
                    dataset_name: str,
                    transform=None,
                    max_samples_per_class: Optional[int] = None,
                    class_subset: Optional[List[str]] = None) -> PlantDiseaseDataset:
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称 ('color', 'grayscale', 'segmented', 'baidu')
            transform: 图像变换
            max_samples_per_class: 每个类别最大样本数
            class_subset: 类别子集
            
        Returns:
            PlantDiseaseDataset实例
        """
        # 获取数据集配置
        if dataset_name in ['color', 'grayscale', 'segmented']:
            dataset_config = self.config.plantvillage_datasets[dataset_name]
        elif dataset_name == 'baidu':
            dataset_config = self.config.baidu_dataset
        else:
            raise ValueError(f"未知的数据集名称: {dataset_name}")
        
        # 创建数据集
        dataset = PlantDiseaseDataset(
            dataset_config=dataset_config,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            class_subset=class_subset
        )
        
        # 缓存数据集
        cache_key = f"{dataset_name}_{max_samples_per_class}_{len(class_subset) if class_subset else 'all'}"
        self.datasets[cache_key] = dataset
        self.dataset_info[cache_key] = dataset.get_dataset_info()
        
        return dataset
    
    def create_combined_dataset(self, 
                              dataset_names: List[str],
                              transform=None,
                              max_samples_per_class: Optional[int] = None) -> PlantDiseaseDataset:
        """
        创建组合数据集
        
        Args:
            dataset_names: 要组合的数据集名称列表
            transform: 图像变换
            max_samples_per_class: 每个类别最大样本数
            
        Returns:
            组合的PlantDiseaseDataset
        """
        # 加载各个数据集
        datasets = []
        for name in dataset_names:
            dataset = self.load_dataset(name, transform, max_samples_per_class)
            datasets.append(dataset)
        
        # 合并数据集（这里简化实现，实际可能需要更复杂的合并逻辑）
        if len(datasets) == 1:
            return datasets[0]
        
        # 创建合并后的数据集
        primary_dataset = datasets[0]
        combined_samples = []
        
        # 收集所有样本
        for dataset in datasets:
            combined_samples.extend(dataset.samples)
        
        # 创建新的数据集实例
        combined_config = primary_dataset.config
        combined_dataset = PlantDiseaseDataset(combined_config, transform)
        combined_dataset.samples = combined_samples
        combined_dataset.classes = primary_dataset.classes
        combined_dataset.class_to_idx = primary_dataset.class_to_idx
        
        return combined_dataset
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取所有数据集的统计信息"""
        stats = {
            'available_datasets': list(self.config.get_all_image_datasets().keys()),
            'path_validation': self.config.validate_paths(),
            'loaded_datasets': list(self.dataset_info.keys()),
            'dataset_details': self.dataset_info
        }
        return stats
    
    def create_data_loaders(self,
                          dataset: PlantDiseaseDataset,
                          batch_size: int = 32,
                          train_split: float = 0.8,
                          shuffle: bool = True,
                          num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """
        创建训练和验证数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批大小
            train_split: 训练集比例
            shuffle: 是否打乱
            num_workers: 工作进程数
            
        Returns:
            (train_loader, val_loader)
        """
        # 分割数据集
        total_size = len(dataset)
        train_size = int(total_size * train_split)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader

# 全局数据集管理器实例
dataset_manager = DatasetManager()

def get_dataset_manager() -> DatasetManager:
    """获取全局数据集管理器实例"""
    return dataset_manager

if __name__ == "__main__":
    # 测试数据集管理器
    print("🧪 数据集管理器测试")
    print("=" * 50)
    
    if not PIL_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    manager = get_dataset_manager()
    
    # 获取统计信息
    stats = manager.get_dataset_statistics()
    print("📊 数据集统计:")
    print(f"   可用数据集: {stats['available_datasets']}")
    print(f"   路径验证: {stats['path_validation']}")
    
    # 测试加载小样本数据集
    try:
        print(f"\n🔍 测试加载color数据集 (每类最多10个样本)...")
        dataset = manager.load_dataset('color', max_samples_per_class=10)
        info = dataset.get_dataset_info()
        
        print(f"✅ 加载成功:")
        print(f"   数据集: {info['name']}")
        print(f"   样本数: {info['total_samples']}")
        print(f"   类别数: {info['num_classes']}")
        print(f"   前5个类别: {list(info['classes'])[:5]}")
        
        # 测试获取单个样本
        if len(dataset) > 0:
            sample_image, sample_label = dataset[0]
            print(f"   样本形状: {sample_image.size if hasattr(sample_image, 'size') else 'N/A'}")
            print(f"   样本标签: {dataset.classes[sample_label]}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print(f"\n✅ 数据集管理器测试完成")