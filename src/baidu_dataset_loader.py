#!/usr/bin/env python3
"""
百度AI Studio数据集加载器
实现ai_challenger_pdr2018数据集的加载器，包含与PlantVillage的类别映射
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
from src.plantvillage_loader import PlantVillageClassMapping

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaiduAIStudioClassMapping:
    """百度AI Studio数据集类别映射管理"""
    
    def __init__(self):
        """初始化类别映射"""
        # 百度AI Studio数据集的数字ID到中文类别名称的映射
        # 基于AI Challenger 2018植物病害识别数据集的61个类别
        self.id_to_chinese = {
            0: '苹果健康',
            1: '苹果黑星病',
            2: '苹果黑腐病',
            3: '苹果雪松锈病',
            4: '樱桃白粉病',
            5: '樱桃健康',
            6: '玉米灰斑病',
            7: '玉米普通锈病',
            8: '玉米北方叶枯病',
            9: '玉米健康',
            10: '葡萄黑腐病',
            11: '葡萄黑麻疹病',
            12: '葡萄叶枯病',
            13: '葡萄健康',
            14: '桃细菌性斑点病',
            15: '桃健康',
            16: '辣椒细菌性斑点病',
            17: '辣椒健康',
            18: '马铃薯早疫病',
            19: '马铃薯晚疫病',
            20: '马铃薯健康',
            21: '覆盆子健康',
            22: '大豆健康',
            23: '南瓜白粉病',
            24: '草莓叶焦病',
            25: '草莓健康',
            26: '番茄细菌性斑点病',
            27: '番茄早疫病',
            28: '番茄晚疫病',
            29: '番茄叶霉病',
            30: '番茄斑点病',
            31: '番茄红蜘蛛',
            32: '番茄靶斑病',
            33: '番茄黄化曲叶病毒',
            34: '番茄花叶病毒',
            35: '番茄健康',
            36: '橙子黄龙病',
            37: '蓝莓健康',
            38: '水稻稻瘟病',
            39: '水稻褐斑病',
            40: '水稻健康',
            41: '小麦条纹花叶病',
            42: '小麦叶锈病',
            43: '小麦健康',
            44: '棉花细菌性疫病',
            45: '棉花健康',
            46: '茄子健康',
            47: '茄子细菌性斑点病',
            48: '黄瓜霜霉病',
            49: '黄瓜健康',
            50: '豆角锈病',
            51: '豆角健康',
            52: '白菜软腐病',
            53: '白菜健康',
            54: '萝卜黑腐病',
            55: '萝卜健康',
            56: '花生叶斑病',
            57: '花生健康',
            58: '向日葵锈病',
            59: '向日葵健康',
            60: '其他病害'
        }
        
        # 中文类别到PlantVillage英文类别的映射
        self.chinese_to_plantvillage = {
            # 苹果类
            '苹果健康': 'Apple___healthy',
            '苹果黑星病': 'Apple___Apple_scab',
            '苹果黑腐病': 'Apple___Black_rot',
            '苹果雪松锈病': 'Apple___Cedar_apple_rust',
            
            # 樱桃类
            '樱桃白粉病': 'Cherry_(including_sour)___Powdery_mildew',
            '樱桃健康': 'Cherry_(including_sour)___healthy',
            
            # 玉米类
            '玉米灰斑病': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            '玉米普通锈病': 'Corn_(maize)___Common_rust_',
            '玉米北方叶枯病': 'Corn_(maize)___Northern_Leaf_Blight',
            '玉米健康': 'Corn_(maize)___healthy',
            
            # 葡萄类
            '葡萄黑腐病': 'Grape___Black_rot',
            '葡萄黑麻疹病': 'Grape___Esca_(Black_Measles)',
            '葡萄叶枯病': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            '葡萄健康': 'Grape___healthy',
            
            # 桃子类
            '桃细菌性斑点病': 'Peach___Bacterial_spot',
            '桃健康': 'Peach___healthy',
            
            # 辣椒类
            '辣椒细菌性斑点病': 'Pepper,_bell___Bacterial_spot',
            '辣椒健康': 'Pepper,_bell___healthy',
            
            # 马铃薯类
            '马铃薯早疫病': 'Potato___Early_blight',
            '马铃薯晚疫病': 'Potato___Late_blight',
            '马铃薯健康': 'Potato___healthy',
            
            # 覆盆子类
            '覆盆子健康': 'Raspberry___healthy',
            
            # 大豆类
            '大豆健康': 'Soybean___healthy',
            
            # 南瓜类
            '南瓜白粉病': 'Squash___Powdery_mildew',
            
            # 草莓类
            '草莓叶焦病': 'Strawberry___Leaf_scorch',
            '草莓健康': 'Strawberry___healthy',
            
            # 番茄类
            '番茄细菌性斑点病': 'Tomato___Bacterial_spot',
            '番茄早疫病': 'Tomato___Early_blight',
            '番茄晚疫病': 'Tomato___Late_blight',
            '番茄叶霉病': 'Tomato___Leaf_Mold',
            '番茄斑点病': 'Tomato___Septoria_leaf_spot',
            '番茄红蜘蛛': 'Tomato___Spider_mites Two-spotted_spider_mite',
            '番茄靶斑病': 'Tomato___Target_Spot',
            '番茄黄化曲叶病毒': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            '番茄花叶病毒': 'Tomato___Tomato_mosaic_virus',
            '番茄健康': 'Tomato___healthy',
            
            # 橙子类
            '橙子黄龙病': 'Orange___Haunglongbing_(Citrus_greening)',
            
            # 蓝莓类
            '蓝莓健康': 'Blueberry___healthy',
            
            # 以下类别在PlantVillage中没有对应类别
            '水稻稻瘟病': None,
            '水稻褐斑病': None,
            '水稻健康': None,
            '小麦条纹花叶病': None,
            '小麦叶锈病': None,
            '小麦健康': None,
            '棉花细菌性疫病': None,
            '棉花健康': None,
            '茄子健康': None,
            '茄子细菌性斑点病': None,
            '黄瓜霜霉病': None,
            '黄瓜健康': None,
            '豆角锈病': None,
            '豆角健康': None,
            '白菜软腐病': None,
            '白菜健康': None,
            '萝卜黑腐病': None,
            '萝卜健康': None,
            '花生叶斑病': None,
            '花生健康': None,
            '向日葵锈病': None,
            '向日葵健康': None,
            '其他病害': None
        }
        
        # 创建反向映射
        self.plantvillage_to_chinese = {v: k for k, v in self.chinese_to_plantvillage.items() if v is not None}
        
        # 获取PlantVillage映射用于中文显示
        self.plantvillage_mapping = PlantVillageClassMapping()
        
        # 百度数据集特有的类别（PlantVillage中没有的）
        self.baidu_unique_classes = [k for k, v in self.chinese_to_plantvillage.items() if v is None]
        
        # 共同类别
        self.common_classes = [k for k, v in self.chinese_to_plantvillage.items() if v is not None]
    
    def get_chinese_name(self, class_id: int) -> str:
        """根据数字ID获取中文类别名称"""
        return self.id_to_chinese.get(class_id, f'未知类别_{class_id}')
    
    def get_class_id(self, chinese_name: str) -> Optional[int]:
        """根据中文名称获取数字ID"""
        for class_id, name in self.id_to_chinese.items():
            if name == chinese_name:
                return class_id
        return None
    
    def get_plantvillage_class(self, class_id: int) -> Optional[str]:
        """根据数字ID获取对应的PlantVillage类别"""
        chinese_name = self.get_chinese_name(class_id)
        return self.chinese_to_plantvillage.get(chinese_name)
    
    def get_plantvillage_class_from_chinese(self, chinese_name: str) -> Optional[str]:
        """根据中文名称获取对应的PlantVillage类别"""
        return self.chinese_to_plantvillage.get(chinese_name)
    
    def is_common_class(self, class_id: int) -> bool:
        """判断是否为与PlantVillage共同的类别"""
        chinese_name = self.get_chinese_name(class_id)
        return chinese_name in self.common_classes
    
    def is_baidu_unique(self, class_id: int) -> bool:
        """判断是否为百度独有类别"""
        chinese_name = self.get_chinese_name(class_id)
        return chinese_name in self.baidu_unique_classes
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """获取映射统计信息"""
        return {
            'total_baidu_classes': len(self.id_to_chinese),
            'common_classes': len(self.common_classes),
            'baidu_unique_classes': len(self.baidu_unique_classes),
            'common_class_list': self.common_classes,
            'baidu_unique_list': self.baidu_unique_classes,
            'mapping_coverage': len(self.common_classes) / len(self.id_to_chinese),
            'class_id_range': f"0-{max(self.id_to_chinese.keys())}"
        }

class BaiduAIStudioDatasetLoader:
    """百度AI Studio数据集专用加载器"""
    
    def __init__(self):
        """初始化加载器"""
        self.config = get_dataset_config()
        self.dataset_config = self.config.baidu_dataset
        self.class_mapping = BaiduAIStudioClassMapping()
        
        # 数据集信息
        self.dataset_info = None
        self.annotation_data = None
        self.class_distribution = None
        
        logger.info("百度AI Studio数据集加载器初始化完成")
    
    def load_annotations(self) -> Dict[str, Any]:
        """加载标注文件"""
        if not os.path.exists(self.dataset_config.path):
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_config.path}")
        
        logger.info("开始加载百度AI Studio标注文件...")
        
        # 查找所有JSON文件
        json_files = []
        txt_files = []
        
        for root, dirs, files in os.walk(self.dataset_config.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.json'):
                    json_files.append(file_path)
                elif file.endswith('.txt') and 'README' not in file:
                    txt_files.append(file_path)
        
        logger.info(f"找到 {len(json_files)} 个JSON文件, {len(txt_files)} 个TXT文件")
        
        # 加载JSON标注
        all_annotations = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_annotations.extend(data)
                    elif isinstance(data, dict):
                        all_annotations.append(data)
                logger.info(f"成功加载 {json_file}")
            except Exception as e:
                logger.warning(f"无法加载JSON文件 {json_file}: {e}")
        
        # 如果没有JSON文件，尝试解析TXT文件
        if not all_annotations and txt_files:
            logger.info("未找到JSON标注，尝试解析TXT文件...")
            all_annotations = self._parse_txt_annotations(txt_files)
        
        # 如果仍然没有标注，尝试从文件名推断
        if not all_annotations:
            logger.info("未找到标注文件，尝试从文件名推断类别...")
            all_annotations = self._infer_from_filenames()
        
        self.annotation_data = all_annotations
        logger.info(f"总共加载了 {len(all_annotations)} 个标注")
        
        return {
            'total_annotations': len(all_annotations),
            'json_files': json_files,
            'txt_files': txt_files,
            'annotations': all_annotations[:5]  # 显示前5个样本
        }
    
    def _parse_txt_annotations(self, txt_files: List[str]) -> List[Dict[str, Any]]:
        """解析TXT标注文件"""
        annotations = []
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 尝试不同的分隔符
                    parts = None
                    for sep in ['\t', ',', ' ', '|']:
                        if sep in line:
                            parts = line.split(sep)
                            break
                    
                    if parts and len(parts) >= 2:
                        image_name = parts[0].strip()
                        label = parts[1].strip()
                        
                        annotations.append({
                            'image': image_name,
                            'label': label,
                            'source': txt_file
                        })
                
                logger.info(f"从 {txt_file} 解析了 {len([a for a in annotations if a.get('source') == txt_file])} 个标注")
                
            except Exception as e:
                logger.warning(f"无法解析TXT文件 {txt_file}: {e}")
        
        return annotations
    
    def _infer_from_filenames(self) -> List[Dict[str, Any]]:
        """从文件名推断类别"""
        annotations = []
        
        # 获取所有图像文件
        image_files = []
        for root, dirs, files in os.walk(self.dataset_config.path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.dataset_config.image_extensions):
                    image_files.append(os.path.join(root, file))
        
        # 尝试从文件名或目录名推断类别
        for img_path in image_files:
            rel_path = os.path.relpath(img_path, self.dataset_config.path)
            
            # 从目录结构推断
            path_parts = rel_path.split(os.sep)
            
            # 假设类别信息在目录名或文件名中
            possible_labels = []
            
            # 检查目录名
            for part in path_parts[:-1]:  # 排除文件名
                if any(keyword in part for keyword in ['train', 'val', 'test', 'images']):
                    continue
                possible_labels.append(part)
            
            # 检查文件名
            filename = os.path.splitext(path_parts[-1])[0]
            if '_' in filename:
                possible_labels.extend(filename.split('_'))
            
            # 选择最可能的标签
            label = 'unknown'
            if possible_labels:
                label = possible_labels[0]  # 简单选择第一个
            
            annotations.append({
                'image': rel_path,
                'label': label,
                'source': 'filename_inference'
            })
        
        logger.info(f"从文件名推断了 {len(annotations)} 个标注")
        return annotations
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """分析数据集结构和内容"""
        if self.annotation_data is None:
            self.load_annotations()
        
        logger.info("开始分析百度AI Studio数据集...")
        
        # 统计类别分布
        class_distribution = Counter()
        chinese_class_distribution = Counter()
        valid_annotations = []
        image_formats = Counter()
        
        # 获取图像目录
        image_dirs = []
        for root, dirs, files in os.walk(self.dataset_config.path):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                image_dirs.append(root)
        
        for ann in self.annotation_data:
            # 获取图像ID和病害类别ID
            image_id = ann.get('image_id', '')
            disease_class = ann.get('disease_class', -1)
            
            if not image_id or disease_class == -1:
                continue
            
            # 查找图像文件
            image_path = None
            for img_dir in image_dirs:
                potential_path = os.path.join(img_dir, image_id)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if not image_path:
                continue
            
            # 获取中文类别名称
            chinese_name = self.class_mapping.get_chinese_name(disease_class)
            plantvillage_class = self.class_mapping.get_plantvillage_class(disease_class)
            
            # 统计格式
            ext = os.path.splitext(image_id)[1].lower()
            image_formats[ext] += 1
            
            # 统计类别
            class_distribution[disease_class] += 1
            chinese_class_distribution[chinese_name] += 1
            
            valid_annotations.append({
                'image_path': image_path,
                'image_id': image_id,
                'disease_class_id': disease_class,
                'chinese_name': chinese_name,
                'plantvillage_class': plantvillage_class,
                'is_common': self.class_mapping.is_common_class(disease_class),
                'is_unique': self.class_mapping.is_baidu_unique(disease_class)
            })
        
        # 创建数据集信息
        self.dataset_info = {
            'dataset_name': 'Baidu_AI_Studio',
            'dataset_path': self.dataset_config.path,
            'total_annotations': len(self.annotation_data),
            'valid_annotations': len(valid_annotations),
            'total_classes': len(class_distribution),
            'class_distribution': dict(class_distribution),
            'chinese_class_distribution': dict(chinese_class_distribution),
            'image_formats': dict(image_formats),
            'mapping_stats': self.class_mapping.get_mapping_statistics(),
            'valid_data': valid_annotations
        }
        
        self.class_distribution = dict(chinese_class_distribution)
        
        logger.info(f"数据集分析完成: {len(class_distribution)} 个类别, {len(valid_annotations)} 个有效标注")
        
        return self.dataset_info
    
    def create_plantvillage_compatible_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """创建与PlantVillage兼容的数据集"""
        if self.dataset_info is None:
            self.analyze_dataset()
        
        logger.info("创建与PlantVillage兼容的数据集...")
        
        compatible_data = {}
        incompatible_data = []
        
        for ann in self.dataset_info['valid_data']:
            if ann['is_common']:
                plantvillage_class = ann['plantvillage_class']
                if plantvillage_class not in compatible_data:
                    compatible_data[plantvillage_class] = []
                
                compatible_data[plantvillage_class].append({
                    'image_path': ann['image_path'],
                    'image_id': ann['image_id'],
                    'disease_class_id': ann['disease_class_id'],
                    'chinese_name': ann['chinese_name'],
                    'plantvillage_label': plantvillage_class,
                    'source': 'baidu'
                })
            else:
                incompatible_data.append(ann)
        
        logger.info(f"兼容数据: {len(compatible_data)} 个PlantVillage类别")
        logger.info(f"不兼容数据: {len(incompatible_data)} 个样本")
        
        return {
            'compatible': compatible_data,
            'incompatible': incompatible_data,
            'stats': {
                'compatible_classes': len(compatible_data),
                'compatible_samples': sum(len(samples) for samples in compatible_data.values()),
                'incompatible_samples': len(incompatible_data)
            }
        }
    
    def create_train_val_split(self, 
                              train_ratio: float = 0.8,
                              random_seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """创建训练/验证集分割"""
        compatible_data = self.create_plantvillage_compatible_dataset()
        
        random.seed(random_seed)
        
        train_split = {}
        val_split = {}
        
        for plantvillage_class, samples in compatible_data['compatible'].items():
            # 随机打乱
            shuffled_samples = samples.copy()
            random.shuffle(shuffled_samples)
            
            # 分割
            split_idx = int(len(shuffled_samples) * train_ratio)
            train_split[plantvillage_class] = shuffled_samples[:split_idx]
            val_split[plantvillage_class] = shuffled_samples[split_idx:]
        
        train_total = sum(len(samples) for samples in train_split.values())
        val_total = sum(len(samples) for samples in val_split.values())
        
        logger.info(f"百度数据集分割完成: 训练集 {train_total} 张, 验证集 {val_total} 张")
        
        return train_split, val_split
    
    def generate_dataset_report(self, output_path: Optional[str] = None) -> str:
        """生成数据集分析报告"""
        if self.dataset_info is None:
            self.analyze_dataset()
        
        compatible_data = self.create_plantvillage_compatible_dataset()
        
        # 生成报告内容
        report_lines = [
            f"# 百度AI Studio数据集分析报告",
            f"",
            f"## 基本信息",
            f"- **数据集名称**: {self.dataset_info['dataset_name']}",
            f"- **数据集路径**: {self.dataset_info['dataset_path']}",
            f"- **总标注数**: {self.dataset_info['total_annotations']}",
            f"- **有效标注数**: {self.dataset_info['valid_annotations']}",
            f"- **总类别数**: {self.dataset_info['total_classes']}",
            f"",
            f"## 与PlantVillage的兼容性分析",
        ]
        
        mapping_stats = self.dataset_info['mapping_stats']
        compat_stats = compatible_data['stats']
        
        report_lines.extend([
            f"- **映射覆盖率**: {mapping_stats['mapping_coverage']:.2%}",
            f"- **共同类别数**: {mapping_stats['common_classes']} / {mapping_stats['total_baidu_classes']}",
            f"- **兼容样本数**: {compat_stats['compatible_samples']}",
            f"- **不兼容样本数**: {compat_stats['incompatible_samples']}",
            f"",
            f"## 类别分布",
            f""
        ])
        
        # 按样本数量排序显示类别
        sorted_classes = sorted(
            self.dataset_info['chinese_class_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        report_lines.append("| 中文类别 | PlantVillage类别 | 样本数量 | 兼容性 |")
        report_lines.append("|----------|------------------|----------|--------|")
        
        for chinese_class, count in sorted_classes:
            pv_class = self.class_mapping.get_plantvillage_class_from_chinese(chinese_class)
            compatibility = "✅ 兼容" if pv_class else "❌ 不兼容"
            pv_display = pv_class if pv_class else "无对应类别"
            
            report_lines.append(
                f"| {chinese_class} | {pv_display} | {count} | {compatibility} |"
            )
        
        # 添加兼容类别详情
        if compatible_data['compatible']:
            report_lines.extend([
                f"",
                f"## 兼容类别详情",
                f""
            ])
            
            for pv_class, samples in compatible_data['compatible'].items():
                chinese_classes = list(set(s['chinese_name'] for s in samples))
                report_lines.append(f"- **{pv_class}**: {len(samples)} 个样本")
                report_lines.append(f"  - 中文类别: {', '.join(chinese_classes)}")
        
        # 添加不兼容类别
        if compatible_data['incompatible']:
            unique_classes = list(set(ann['chinese_name'] for ann in compatible_data['incompatible']))
            report_lines.extend([
                f"",
                f"## 百度独有类别",
                f""
            ])
            
            for unique_class in unique_classes:
                count = sum(1 for ann in compatible_data['incompatible'] if ann['chinese_name'] == unique_class)
                report_lines.append(f"- **{unique_class}**: {count} 个样本")
        
        report_content = "\n".join(report_lines)
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"百度数据集报告已保存到: {output_path}")
        
        return report_content

# 便捷函数
def create_baidu_loader() -> BaiduAIStudioDatasetLoader:
    """创建百度AI Studio数据集加载器"""
    return BaiduAIStudioDatasetLoader()

def get_baidu_class_mapping() -> BaiduAIStudioClassMapping:
    """获取百度AI Studio类别映射"""
    return BaiduAIStudioClassMapping()

if __name__ == "__main__":
    # 测试百度AI Studio加载器
    print("🧪 百度AI Studio数据集加载器测试")
    print("=" * 60)
    
    if not PIL_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试类别映射
        print("📋 测试类别映射...")
        mapping = get_baidu_class_mapping()
        stats = mapping.get_mapping_statistics()
        
        print(f"✅ 类别映射统计:")
        print(f"   总百度类别数: {stats['total_baidu_classes']}")
        print(f"   共同类别数: {stats['common_classes']}")
        print(f"   百度独有类别数: {stats['baidu_unique_classes']}")
        print(f"   映射覆盖率: {stats['mapping_coverage']:.2%}")
        
        # 测试数据集加载器
        print(f"\n🔍 测试百度数据集加载器...")
        loader = create_baidu_loader()
        
        # 加载标注
        ann_info = loader.load_annotations()
        print(f"✅ 标注加载完成:")
        print(f"   总标注数: {ann_info['total_annotations']}")
        print(f"   JSON文件: {len(ann_info['json_files'])}")
        print(f"   TXT文件: {len(ann_info['txt_files'])}")
        
        # 分析数据集
        dataset_info = loader.analyze_dataset()
        print(f"✅ 数据集分析完成:")
        print(f"   有效标注数: {dataset_info['valid_annotations']}")
        print(f"   总类别数: {dataset_info['total_classes']}")
        
        # 测试兼容性分析
        print(f"\n📊 测试兼容性分析...")
        compatible_data = loader.create_plantvillage_compatible_dataset()
        stats = compatible_data['stats']
        
        print(f"✅ 兼容性分析完成:")
        print(f"   兼容类别数: {stats['compatible_classes']}")
        print(f"   兼容样本数: {stats['compatible_samples']}")
        print(f"   不兼容样本数: {stats['incompatible_samples']}")
        
        # 生成报告
        print(f"\n📄 生成数据集报告...")
        report_path = "Baidu_AI_Studio_dataset_report.md"
        report = loader.generate_dataset_report(report_path)
        print(f"✅ 报告已生成: {report_path}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 百度AI Studio数据集加载器测试完成")