#!/usr/bin/env python3
"""
数据集配置文件
定义所有数据集的路径和参数配置
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

@dataclass
class ImageDatasetConfig:
    """图像数据集配置"""
    name: str
    path: str
    description: str
    image_extensions: List[str]
    expected_classes: int
    input_size: Tuple[int, int]
    enabled: bool = True
    has_subdirectories: bool = True

@dataclass
class PhenologyDatasetConfig:
    """物候数据集配置"""
    name: str
    path: str
    description: str
    data_extensions: List[str]
    temporal_range: Tuple[int, int]  # 年份范围
    spatial_resolution: str
    enabled: bool = True

class CropPilotDatasetConfig:
    """CropPilot项目数据集配置管理"""
    
    def __init__(self, base_data_path: str = None):
        """
        初始化数据集配置
        
        Args:
            base_data_path: 数据集基础路径，如果为None则使用默认路径
        """
        if base_data_path is None:
            # 默认数据集路径
            self.base_data_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据"
        else:
            self.base_data_path = base_data_path
        
        self._setup_datasets()
    
    def _setup_datasets(self):
        """设置所有数据集配置"""
        
        # PlantVillage数据集配置
        plantvillage_base = os.path.join(
            self.base_data_path, 
            "1.图像数据（病虫害识别核心）", 
            "plantvillage dataset"
        )
        
        self.plantvillage_datasets = {
            'color': ImageDatasetConfig(
                name="PlantVillage_Color",
                path=os.path.join(plantvillage_base, "color"),
                description="PlantVillage彩色图像数据集",
                image_extensions=['.jpg', '.jpeg', '.png'],
                expected_classes=38,
                input_size=(224, 224),
                has_subdirectories=True
            ),
            'grayscale': ImageDatasetConfig(
                name="PlantVillage_Grayscale", 
                path=os.path.join(plantvillage_base, "grayscale"),
                description="PlantVillage灰度图像数据集",
                image_extensions=['.jpg', '.jpeg', '.png'],
                expected_classes=38,
                input_size=(224, 224),
                has_subdirectories=True
            ),
            'segmented': ImageDatasetConfig(
                name="PlantVillage_Segmented",
                path=os.path.join(plantvillage_base, "segmented"), 
                description="PlantVillage分割图像数据集",
                image_extensions=['.jpg', '.jpeg', '.png'],
                expected_classes=38,
                input_size=(224, 224),
                has_subdirectories=True
            )
        }
        
        # 百度AI Studio数据集配置
        self.baidu_dataset = ImageDatasetConfig(
            name="Baidu_AI_Studio",
            path=os.path.join(
                self.base_data_path,
                "1.图像数据（病虫害识别核心）",
                "ai_challenger_pdr2018"
            ),
            description="百度AI Studio植物病害数据集",
            image_extensions=['.jpg', '.jpeg', '.png'],
            expected_classes=0,  # 需要从标注文件中确定
            input_size=(224, 224),
            has_subdirectories=False
        )
        
        # 物候数据集配置
        self.phenology_dataset = PhenologyDatasetConfig(
            name="ChinaCropPhen1km",
            path=os.path.join(
                self.base_data_path,
                "2.生长数据（时间序列）",
                "8313530"
            ),
            description="中国作物物候1km分辨率数据集",
            data_extensions=['.tif', '.tiff', '.nc', '.hdf'],
            temporal_range=(2000, 2019),
            spatial_resolution="1km"
        )
    
    def get_primary_dataset_config(self) -> ImageDatasetConfig:
        """获取主要训练数据集配置（PlantVillage Color用于基础训练）"""
        return self.plantvillage_datasets['color']
    
    def get_validation_dataset_config(self) -> ImageDatasetConfig:
        """获取验证数据集配置（百度数据集用于中国本土化验证）"""
        return self.baidu_dataset
    
    def get_enhancement_dataset_config(self) -> ImageDatasetConfig:
        """获取精度增强数据集配置（Segmented用于精度优化）"""
        return self.plantvillage_datasets['segmented']
    
    def get_all_image_datasets(self) -> Dict[str, ImageDatasetConfig]:
        """获取所有图像数据集配置"""
        datasets = {}
        datasets.update(self.plantvillage_datasets)
        datasets['baidu'] = self.baidu_dataset
        return datasets
    
    def get_phenology_config(self) -> PhenologyDatasetConfig:
        """获取物候数据集配置"""
        return self.phenology_dataset
    
    def validate_paths(self) -> Dict[str, bool]:
        """验证所有数据集路径是否存在"""
        results = {}
        
        # 检查PlantVillage数据集
        for name, config in self.plantvillage_datasets.items():
            results[f"plantvillage_{name}"] = os.path.exists(config.path)
        
        # 检查百度数据集
        results["baidu"] = os.path.exists(self.baidu_dataset.path)
        
        # 检查物候数据集
        results["phenology"] = os.path.exists(self.phenology_dataset.path)
        
        return results
    
    def get_dataset_summary(self) -> Dict[str, any]:
        """获取数据集配置摘要"""
        path_status = self.validate_paths()
        
        return {
            'base_path': self.base_data_path,
            'plantvillage_datasets': len(self.plantvillage_datasets),
            'total_image_datasets': len(self.get_all_image_datasets()),
            'has_phenology_data': path_status.get('phenology', False),
            'path_validation': path_status,
            'training_strategy': {
                'primary_training': self.get_primary_dataset_config().name,
                'localization_validation': self.get_validation_dataset_config().name,
                'precision_enhancement': self.get_enhancement_dataset_config().name,
                'context_data': self.get_phenology_config().name
            },
            'recommended_workflow': [
                "阶段1: PlantVillage Color基础训练",
                "阶段2: PlantVillage Segmented精度优化", 
                "阶段3: 百度数据集中国本土化微调",
                "阶段4: 物候数据上下文增强"
            ]
        }

# 全局配置实例
dataset_config = CropPilotDatasetConfig()

# 便捷访问函数
def get_dataset_config() -> CropPilotDatasetConfig:
    """获取全局数据集配置实例"""
    return dataset_config

def get_primary_dataset_path() -> str:
    """获取主要训练数据集路径"""
    return dataset_config.get_primary_dataset_config().path

def get_validation_dataset_path() -> str:
    """获取验证数据集路径"""
    return dataset_config.get_validation_dataset_config().path

if __name__ == "__main__":
    # 测试配置
    print("🔧 CropPilot 数据集配置测试")
    print("=" * 50)
    
    config = get_dataset_config()
    summary = config.get_dataset_summary()
    
    print(f"📁 基础路径: {summary['base_path']}")
    print(f"📊 PlantVillage数据集: {summary['plantvillage_datasets']} 个")
    print(f"📊 总图像数据集: {summary['total_image_datasets']} 个")
    print(f"🌱 物候数据: {'✅' if summary['has_phenology_data'] else '❌'}")
    print(f"🎯 训练策略:")
    for stage in summary['recommended_workflow']:
        print(f"   {stage}")
    
    print(f"\n📊 数据集角色:")
    strategy = summary['training_strategy']
    print(f"   主要训练: {strategy['primary_training']}")
    print(f"   本土化验证: {strategy['localization_validation']}")
    print(f"   精度增强: {strategy['precision_enhancement']}")
    print(f"   上下文数据: {strategy['context_data']}")
    
    print(f"\n📋 路径验证结果:")
    for dataset_name, exists in summary['path_validation'].items():
        status = "✅" if exists else "❌"
        print(f"   {dataset_name}: {status}")
    
    # 显示主要数据集详情
    primary_config = config.get_primary_dataset_config()
    print(f"\n🎯 主要数据集详情:")
    print(f"   名称: {primary_config.name}")
    print(f"   路径: {primary_config.path}")
    print(f"   描述: {primary_config.description}")
    print(f"   预期类别: {primary_config.expected_classes}")
    print(f"   输入尺寸: {primary_config.input_size}")