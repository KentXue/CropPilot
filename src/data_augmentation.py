#!/usr/bin/env python3
"""
植物病害数据增强策略
实现针对植物病害识别的专业数据增强方法
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from enum import Enum
import random
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import torch
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentationStrategy(Enum):
    """数据增强策略枚举"""
    LIGHT = "light"          # 轻度增强
    MODERATE = "moderate"    # 中度增强
    HEAVY = "heavy"         # 重度增强
    DISEASE_SPECIFIC = "disease_specific"  # 病害特定增强

class PlantDiseaseAugmentation:
    """植物病害专用数据增强器"""
    
    def __init__(self, strategy: AugmentationStrategy = AugmentationStrategy.MODERATE):
        """
        初始化数据增强器
        
        Args:
            strategy: 增强策略
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("需要安装必要依赖")
        
        self.strategy = strategy
        self.augmentation_configs = self._setup_augmentation_configs()
        
        logger.info(f"植物病害数据增强器初始化完成 - 策略: {strategy.value}")
    
    def _setup_augmentation_configs(self) -> Dict[str, Dict[str, Any]]:
        """设置不同增强策略的配置"""
        return {
            AugmentationStrategy.LIGHT.value: {
                'geometric_prob': 0.3,
                'color_prob': 0.2,
                'noise_prob': 0.1,
                'weather_prob': 0.05,
                'occlusion_prob': 0.1
            },
            AugmentationStrategy.MODERATE.value: {
                'geometric_prob': 0.5,
                'color_prob': 0.4,
                'noise_prob': 0.2,
                'weather_prob': 0.15,
                'occlusion_prob': 0.2
            },
            AugmentationStrategy.HEAVY.value: {
                'geometric_prob': 0.7,
                'color_prob': 0.6,
                'noise_prob': 0.3,
                'weather_prob': 0.25,
                'occlusion_prob': 0.3
            },
            AugmentationStrategy.DISEASE_SPECIFIC.value: {
                'geometric_prob': 0.4,
                'color_prob': 0.5,
                'noise_prob': 0.2,
                'weather_prob': 0.2,
                'occlusion_prob': 0.25,
                'disease_simulation_prob': 0.3
            }
        }
    
    def create_leaf_mask(self, image: np.ndarray) -> np.ndarray:
        """
        创建叶片掩码（简化实现）
        
        Args:
            image: 输入图像
            
        Returns:
            叶片掩码
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 定义绿色范围（叶片通常是绿色的）
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # 创建绿色掩码
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        return green_mask
    
    def simulate_disease_spots(self, image: np.ndarray, num_spots: int = None) -> np.ndarray:
        """
        模拟病害斑点
        
        Args:
            image: 输入图像
            num_spots: 斑点数量，如果为None则随机生成
            
        Returns:
            添加病害斑点的图像
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        if num_spots is None:
            num_spots = random.randint(3, 8)
        
        # 创建叶片掩码
        leaf_mask = self.create_leaf_mask(image)
        
        for _ in range(num_spots):
            # 在叶片区域随机选择位置
            leaf_pixels = np.where(leaf_mask > 0)
            if len(leaf_pixels[0]) == 0:
                continue
            
            idx = random.randint(0, len(leaf_pixels[0]) - 1)
            center_y, center_x = leaf_pixels[0][idx], leaf_pixels[1][idx]
            
            # 随机斑点大小和颜色
            radius = random.randint(5, 20)
            
            # 病害斑点通常是褐色、黄色或黑色
            spot_colors = [
                (139, 69, 19),   # 褐色
                (255, 255, 0),   # 黄色
                (50, 50, 50),    # 深灰色
                (160, 82, 45),   # 棕色
            ]
            spot_color = random.choice(spot_colors)
            
            # 绘制斑点
            cv2.circle(result, (center_x, center_y), radius, spot_color, -1)
            
            # 添加边缘模糊效果
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # 高斯模糊边缘
            blurred = cv2.GaussianBlur(result, (15, 15), 0)
            result = np.where(mask[..., None] > 0, 
                            0.7 * result + 0.3 * blurred, 
                            result)
        
        return result.astype(np.uint8)
    
    def simulate_leaf_yellowing(self, image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        模拟叶片黄化
        
        Args:
            image: 输入图像
            intensity: 黄化强度 (0-1)
            
        Returns:
            黄化处理后的图像
        """
        result = image.copy().astype(np.float32)
        
        # 创建叶片掩码
        leaf_mask = self.create_leaf_mask(image)
        
        # 增加黄色通道，减少绿色通道
        yellow_effect = np.zeros_like(result)
        yellow_effect[:, :, 0] = 255  # 红色
        yellow_effect[:, :, 1] = 255  # 绿色
        yellow_effect[:, :, 2] = 0    # 蓝色
        
        # 只在叶片区域应用黄化效果
        mask_3d = np.stack([leaf_mask, leaf_mask, leaf_mask], axis=2) / 255.0
        result = result * (1 - intensity * mask_3d) + yellow_effect * intensity * mask_3d
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def simulate_leaf_wilting(self, image: np.ndarray) -> np.ndarray:
        """
        模拟叶片萎蔫（通过几何变形）
        
        Args:
            image: 输入图像
            
        Returns:
            萎蔫效果图像
        """
        h, w = image.shape[:2]
        
        # 创建随机变形场
        displacement_x = np.random.normal(0, 2, (h, w)).astype(np.float32)
        displacement_y = np.random.normal(0, 2, (h, w)).astype(np.float32)
        
        # 创建映射矩阵
        map_x = np.arange(w, dtype=np.float32).reshape(1, -1).repeat(h, axis=0)
        map_y = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, axis=1)
        
        map_x += displacement_x
        map_y += displacement_y
        
        # 应用变形
        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result
    
    def simulate_insect_damage(self, image: np.ndarray, num_holes: int = None) -> np.ndarray:
        """
        模拟虫害（叶片上的洞）
        
        Args:
            image: 输入图像
            num_holes: 洞的数量
            
        Returns:
            虫害效果图像
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        if num_holes is None:
            num_holes = random.randint(2, 6)
        
        # 创建叶片掩码
        leaf_mask = self.create_leaf_mask(image)
        
        for _ in range(num_holes):
            # 在叶片区域随机选择位置
            leaf_pixels = np.where(leaf_mask > 0)
            if len(leaf_pixels[0]) == 0:
                continue
            
            idx = random.randint(0, len(leaf_pixels[0]) - 1)
            center_y, center_x = leaf_pixels[0][idx], leaf_pixels[1][idx]
            
            # 随机洞的大小
            hole_size = random.randint(3, 12)
            
            # 创建不规则形状的洞
            angles = np.linspace(0, 2*np.pi, 8)
            radii = np.random.uniform(0.5, 1.5, 8) * hole_size
            
            points = []
            for angle, radius in zip(angles, radii):
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                points.append([x, y])
            
            # 填充洞（使用背景色或透明）
            cv2.fillPoly(result, [np.array(points)], (0, 0, 0))
        
        return result
    
    def create_disease_specific_augmentation(self, disease_type: str) -> A.Compose:
        """
        创建病害特定的数据增强
        
        Args:
            disease_type: 病害类型
            
        Returns:
            病害特定的增强管道
        """
        config = self.augmentation_configs[self.strategy.value]
        
        # 基础增强
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=config['geometric_prob']),
        ]
        
        # 根据病害类型添加特定增强
        if 'spot' in disease_type.lower() or '斑点' in disease_type:
            # 斑点病害：增加对比度和锐化
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.3,
                    p=config['color_prob']
                ),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            ])
        
        elif 'rust' in disease_type.lower() or '锈病' in disease_type:
            # 锈病：增加橙黄色调
            transforms.extend([
                A.HueSaturationValue(
                    hue_shift_limit=(-10, 20),
                    sat_shift_limit=10,
                    val_shift_limit=10,
                    p=config['color_prob']
                ),
            ])
        
        elif 'blight' in disease_type.lower() or '疫病' in disease_type:
            # 疫病：模拟水渍状病斑
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.1),
                    contrast_limit=0.2,
                    p=config['color_prob']
                ),
                A.GaussNoise(var_limit=(10.0, 30.0), p=config['noise_prob']),
            ])
        
        elif 'mildew' in disease_type.lower() or '白粉病' in disease_type:
            # 白粉病：增加白色粉状效果
            transforms.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(-0.1, 0.1),
                    p=config['color_prob']
                ),
                A.GaussNoise(var_limit=(5.0, 20.0), p=config['noise_prob']),
            ])
        
        # 添加通用增强
        transforms.extend([
            A.CoarseDropout(
                max_holes=2,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=config['occlusion_prob']
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def create_background_separation_augmentation(self) -> A.Compose:
        """创建背景分离增强"""
        return A.Compose([
            # 背景替换
            A.RandomCrop(height=200, width=200, p=0.3),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # 主体提取增强
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.4),
            A.RandomBrightnessContrast(p=0.3),
            
            # 背景模糊
            A.MotionBlur(blur_limit=7, p=0.2),
            
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def create_multi_scale_augmentation(self, scales: List[int] = None) -> List[A.Compose]:
        """
        创建多尺度增强
        
        Args:
            scales: 尺度列表
            
        Returns:
            多尺度增强管道列表
        """
        if scales is None:
            scales = [224, 256, 288, 320]
        
        augmentations = []
        config = self.augmentation_configs[self.strategy.value]
        
        for scale in scales:
            aug = A.Compose([
                A.Resize(height=scale, width=scale),
                A.RandomCrop(height=224, width=224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=config['color_prob']),
                A.HueSaturationValue(p=config['color_prob']),
                A.GaussNoise(var_limit=(5.0, 25.0), p=config['noise_prob']),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            augmentations.append(aug)
        
        return augmentations
    
    def visualize_augmentation_effects(self, 
                                     image: np.ndarray,
                                     save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        可视化增强效果
        
        Args:
            image: 输入图像
            save_path: 保存路径
            
        Returns:
            增强效果字典
        """
        effects = {}
        
        # 原图
        effects['original'] = image.copy()
        
        # 病害斑点模拟
        effects['disease_spots'] = self.simulate_disease_spots(image)
        
        # 叶片黄化
        effects['yellowing'] = self.simulate_leaf_yellowing(image)
        
        # 叶片萎蔫
        effects['wilting'] = self.simulate_leaf_wilting(image)
        
        # 虫害
        effects['insect_damage'] = self.simulate_insect_damage(image)
        
        # 如果指定保存路径，创建对比图
        if save_path:
            self._save_comparison_image(effects, save_path)
        
        return effects
    
    def _save_comparison_image(self, effects: Dict[str, np.ndarray], save_path: str):
        """保存对比图像"""
        # 创建2x3的网格
        rows, cols = 2, 3
        effect_names = list(effects.keys())[:6]  # 最多显示6个效果
        
        if not effect_names:
            return
        
        # 获取图像尺寸
        h, w = effects[effect_names[0]].shape[:2]
        
        # 创建大图
        grid_image = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        
        for i, effect_name in enumerate(effect_names):
            row = i // cols
            col = i % cols
            
            start_y = row * h
            end_y = start_y + h
            start_x = col * w
            end_x = start_x + w
            
            grid_image[start_y:end_y, start_x:end_x] = effects[effect_name]
            
            # 添加标题
            cv2.putText(grid_image, effect_name, 
                       (start_x + 10, start_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存图像
        cv2.imwrite(save_path, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))

class AugmentationVisualizer:
    """数据增强可视化工具"""
    
    def __init__(self):
        """初始化可视化工具"""
        self.augmenter = PlantDiseaseAugmentation()
    
    def create_augmentation_gallery(self, 
                                  image_paths: List[str],
                                  output_dir: str,
                                  num_samples: int = 5) -> None:
        """
        创建增强效果画廊
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            num_samples: 样本数量
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 随机选择样本
        selected_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
        
        for i, img_path in enumerate(selected_paths):
            # 加载图像
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 生成增强效果
            save_path = os.path.join(output_dir, f"augmentation_effects_{i+1}.jpg")
            self.augmenter.visualize_augmentation_effects(image, save_path)
            
            logger.info(f"增强效果已保存: {save_path}")

# 便捷函数
def create_plant_augmenter(strategy: AugmentationStrategy = AugmentationStrategy.MODERATE) -> PlantDiseaseAugmentation:
    """创建植物病害数据增强器"""
    return PlantDiseaseAugmentation(strategy)

def create_augmentation_visualizer() -> AugmentationVisualizer:
    """创建增强可视化工具"""
    return AugmentationVisualizer()

if __name__ == "__main__":
    # 测试数据增强模块
    print("🧪 植物病害数据增强测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试增强器创建
        print("📋 测试增强器创建...")
        
        augmenter = create_plant_augmenter(AugmentationStrategy.MODERATE)
        print(f"✅ 增强器创建成功 - 策略: {augmenter.strategy.value}")
        
        # 创建测试图像
        print(f"\n🔍 测试增强效果...")
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 测试病害斑点模拟
        spotted_image = augmenter.simulate_disease_spots(test_image)
        print(f"✅ 病害斑点模拟完成")
        
        # 测试叶片黄化
        yellowed_image = augmenter.simulate_leaf_yellowing(test_image)
        print(f"✅ 叶片黄化模拟完成")
        
        # 测试虫害模拟
        damaged_image = augmenter.simulate_insect_damage(test_image)
        print(f"✅ 虫害模拟完成")
        
        # 测试病害特定增强
        print(f"\n🎯 测试病害特定增强...")
        spot_aug = augmenter.create_disease_specific_augmentation("斑点病")
        rust_aug = augmenter.create_disease_specific_augmentation("锈病")
        
        print(f"✅ 病害特定增强创建完成:")
        print(f"   斑点病增强管道: {len(spot_aug.transforms)} 个变换")
        print(f"   锈病增强管道: {len(rust_aug.transforms)} 个变换")
        
        # 测试多尺度增强
        print(f"\n📏 测试多尺度增强...")
        multi_scale_augs = augmenter.create_multi_scale_augmentation()
        print(f"✅ 多尺度增强创建完成: {len(multi_scale_augs)} 个尺度")
        
        # 测试背景分离增强
        print(f"\n🎭 测试背景分离增强...")
        bg_aug = augmenter.create_background_separation_augmentation()
        print(f"✅ 背景分离增强创建完成: {len(bg_aug.transforms)} 个变换")
        
        # 测试可视化
        print(f"\n🎨 测试增强效果可视化...")
        effects = augmenter.visualize_augmentation_effects(test_image)
        print(f"✅ 可视化完成: {len(effects)} 种效果")
        print(f"   效果类型: {list(effects.keys())}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 植物病害数据增强测试完成")