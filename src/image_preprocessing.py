#!/usr/bin/env python3
"""
高级图像预处理管道
使用Albumentations实现植物病害图像的专业预处理和数据增强
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from enum import Enum

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import torch
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    ALBUMENTATIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    ALBUMENTATIONS_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingMode(Enum):
    """预处理模式枚举"""
    TRAINING = "training"
    VALIDATION = "validation"
    INFERENCE = "inference"
    QUALITY_CHECK = "quality_check"

class ImageQualityAssessment:
    """图像质量评估器"""
    
    def __init__(self):
        """初始化质量评估器"""
        self.quality_thresholds = {
            'min_resolution': (64, 64),
            'max_resolution': (4096, 4096),
            'min_brightness': 20,
            'max_brightness': 235,
            'min_contrast': 0.1,
            'max_blur_variance': 100,  # Laplacian方差阈值
            'min_saturation': 0.05,
            'max_noise_level': 0.3
        }
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        评估图像质量
        
        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            
        Returns:
            质量评估结果字典
        """
        if len(image.shape) == 3:
            height, width, channels = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if channels == 3 else image[:,:,0]
        else:
            height, width = image.shape
            gray = image
            channels = 1
        
        # 基本尺寸检查
        resolution_ok = (
            width >= self.quality_thresholds['min_resolution'][0] and
            height >= self.quality_thresholds['min_resolution'][1] and
            width <= self.quality_thresholds['max_resolution'][0] and
            height <= self.quality_thresholds['max_resolution'][1]
        )
        
        # 亮度检查
        mean_brightness = np.mean(gray)
        brightness_ok = (
            self.quality_thresholds['min_brightness'] <= mean_brightness <= 
            self.quality_thresholds['max_brightness']
        )
        
        # 对比度检查
        contrast = np.std(gray) / 255.0
        contrast_ok = contrast >= self.quality_thresholds['min_contrast']
        
        # 模糊检查（使用Laplacian方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_ok = laplacian_var >= self.quality_thresholds['max_blur_variance']
        
        # 饱和度检查（仅对彩色图像）
        saturation_ok = True
        if len(image.shape) == 3 and channels == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:,:,1]) / 255.0
            saturation_ok = saturation >= self.quality_thresholds['min_saturation']
        
        # 噪声检查（简化实现）
        noise_level = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        noise_ok = noise_level <= self.quality_thresholds['max_noise_level']
        
        # 综合质量评分
        quality_checks = [resolution_ok, brightness_ok, contrast_ok, blur_ok, saturation_ok, noise_ok]
        quality_score = sum(quality_checks) / len(quality_checks)
        
        return {
            'overall_quality': quality_score,
            'is_good_quality': quality_score >= 0.7,
            'resolution': (width, height),
            'resolution_ok': resolution_ok,
            'brightness': mean_brightness,
            'brightness_ok': brightness_ok,
            'contrast': contrast,
            'contrast_ok': contrast_ok,
            'blur_variance': laplacian_var,
            'blur_ok': blur_ok,
            'saturation_ok': saturation_ok,
            'noise_level': noise_level,
            'noise_ok': noise_ok,
            'issues': self._identify_issues(
                resolution_ok, brightness_ok, contrast_ok, 
                blur_ok, saturation_ok, noise_ok
            )
        }
    
    def _identify_issues(self, resolution_ok: bool, brightness_ok: bool, 
                        contrast_ok: bool, blur_ok: bool, 
                        saturation_ok: bool, noise_ok: bool) -> List[str]:
        """识别图像质量问题"""
        issues = []
        if not resolution_ok:
            issues.append("分辨率异常")
        if not brightness_ok:
            issues.append("亮度异常")
        if not contrast_ok:
            issues.append("对比度过低")
        if not blur_ok:
            issues.append("图像模糊")
        if not saturation_ok:
            issues.append("饱和度过低")
        if not noise_ok:
            issues.append("噪声过多")
        return issues

class PlantDiseasePreprocessor:
    """植物病害图像预处理器"""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 mode: PreprocessingMode = PreprocessingMode.TRAINING):
        """
        初始化预处理器
        
        Args:
            input_size: 目标图像尺寸
            mode: 预处理模式
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("需要安装albumentations库: pip install albumentations")
        
        self.input_size = input_size
        self.mode = mode
        self.quality_assessor = ImageQualityAssessment()
        
        # 创建预处理管道
        self.transforms = self._create_transforms()
        
        logger.info(f"植物病害预处理器初始化完成 - 模式: {mode.value}, 尺寸: {input_size}")
    
    def _create_transforms(self) -> A.Compose:
        """创建Albumentations变换管道"""
        
        if self.mode == PreprocessingMode.TRAINING:
            return self._create_training_transforms()
        elif self.mode == PreprocessingMode.VALIDATION:
            return self._create_validation_transforms()
        elif self.mode == PreprocessingMode.INFERENCE:
            return self._create_inference_transforms()
        else:  # QUALITY_CHECK
            return self._create_quality_check_transforms()
    
    def _create_training_transforms(self) -> A.Compose:
        """创建训练时的数据增强管道"""
        return A.Compose([
            # 基础几何变换
            A.Resize(height=self.input_size[0] + 32, width=self.input_size[1] + 32),
            A.RandomCrop(height=self.input_size[0], width=self.input_size[1]),
            
            # 植物图像专用的几何增强
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),  # 植物可能倒置生长
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.4),  # 小角度旋转
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-10, 10),
                p=0.3
            ),
            
            # 光照和颜色增强（模拟不同拍摄条件）
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.4
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.3
            ),
            
            # 模拟不同光照条件
            A.RandomShadow(p=0.2),
            
            # 模拟相机效果
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.MotionBlur(blur_limit=3, p=0.1),
            
            # 裁剪和遮挡（模拟部分遮挡的叶片）
            A.CoarseDropout(
                max_holes=3,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.2
            ),
            
            # 最终标准化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def _create_validation_transforms(self) -> A.Compose:
        """创建验证时的预处理管道"""
        return A.Compose([
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def _create_inference_transforms(self) -> A.Compose:
        """创建推理时的预处理管道"""
        return A.Compose([
            # 多尺度测试增强
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            
            # 轻微的质量增强
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def _create_quality_check_transforms(self) -> A.Compose:
        """创建质量检查的预处理管道"""
        return A.Compose([
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            # 不进行标准化，保持原始像素值用于质量分析
        ])
    
    def preprocess_image(self, 
                        image: Union[np.ndarray, str, Path],
                        return_quality_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        预处理单张图像
        
        Args:
            image: 输入图像（numpy数组、文件路径或Path对象）
            return_quality_info: 是否返回质量信息
            
        Returns:
            预处理后的图像张量，可选质量信息
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image))
            if image_array is None:
                raise ValueError(f"无法加载图像: {image}")
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_array = image.copy()
        
        # 质量评估
        quality_info = None
        if return_quality_info or self.mode == PreprocessingMode.QUALITY_CHECK:
            quality_info = self.quality_assessor.assess_image_quality(image_array)
            
            # 如果质量太差，可以选择跳过或应用修复
            if not quality_info['is_good_quality']:
                logger.warning(f"图像质量较差: {quality_info['issues']}")
                # 应用质量修复
                image_array = self._apply_quality_fixes(image_array, quality_info)
        
        # 应用变换
        try:
            if self.mode == PreprocessingMode.QUALITY_CHECK:
                # 质量检查模式不应用标准化
                transformed = self.transforms(image=image_array)
                result = transformed['image']
            else:
                transformed = self.transforms(image=image_array)
                result = transformed['image']
            
            if return_quality_info:
                return result, quality_info
            else:
                return result
                
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise
    
    def _apply_quality_fixes(self, image: np.ndarray, quality_info: Dict[str, Any]) -> np.ndarray:
        """应用图像质量修复"""
        fixed_image = image.copy()
        
        # 亮度修复
        if not quality_info['brightness_ok']:
            if quality_info['brightness'] < 50:
                # 图像过暗，增加亮度
                fixed_image = cv2.convertScaleAbs(fixed_image, alpha=1.2, beta=30)
            elif quality_info['brightness'] > 200:
                # 图像过亮，降低亮度
                fixed_image = cv2.convertScaleAbs(fixed_image, alpha=0.8, beta=-20)
        
        # 对比度修复
        if not quality_info['contrast_ok']:
            # 应用CLAHE增强对比度
            if len(fixed_image.shape) == 3:
                lab = cv2.cvtColor(fixed_image, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                fixed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                fixed_image = clahe.apply(fixed_image)
        
        # 噪声修复
        if not quality_info['noise_ok']:
            # 应用双边滤波去噪
            fixed_image = cv2.bilateralFilter(fixed_image, 9, 75, 75)
        
        return fixed_image
    
    def preprocess_batch(self, 
                        images: List[Union[np.ndarray, str, Path]],
                        return_quality_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Any]]]]:
        """
        批量预处理图像
        
        Args:
            images: 图像列表
            return_quality_info: 是否返回质量信息
            
        Returns:
            批量预处理后的图像张量，可选质量信息列表
        """
        processed_images = []
        quality_infos = []
        
        for image in images:
            if return_quality_info:
                processed_img, quality_info = self.preprocess_image(image, return_quality_info=True)
                processed_images.append(processed_img)
                quality_infos.append(quality_info)
            else:
                processed_img = self.preprocess_image(image, return_quality_info=False)
                processed_images.append(processed_img)
        
        # 堆叠为批次张量
        batch_tensor = torch.stack(processed_images)
        
        if return_quality_info:
            return batch_tensor, quality_infos
        else:
            return batch_tensor
    
    def create_test_time_augmentation(self, image: Union[np.ndarray, str, Path]) -> torch.Tensor:
        """
        创建测试时增强（TTA）
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像批次 (N, C, H, W)
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            image_array = image.copy()
        
        # 创建TTA变换
        tta_transforms = [
            A.Compose([A.Resize(*self.input_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),  # 原图
            A.Compose([A.HorizontalFlip(p=1.0), A.Resize(*self.input_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),  # 水平翻转
            A.Compose([A.Rotate(limit=5, p=1.0), A.Resize(*self.input_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),  # 轻微旋转
            A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Resize(*self.input_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),  # 亮度调整
        ]
        
        augmented_images = []
        for transform in tta_transforms:
            augmented = transform(image=image_array)['image']
            augmented_images.append(augmented)
        
        return torch.stack(augmented_images)

class PreprocessingPipeline:
    """预处理管道管理器"""
    
    def __init__(self):
        """初始化管道管理器"""
        self.preprocessors = {}
        self.statistics = {
            'processed_images': 0,
            'quality_issues': 0,
            'processing_times': []
        }
    
    def get_preprocessor(self, 
                        mode: PreprocessingMode,
                        input_size: Tuple[int, int] = (224, 224)) -> PlantDiseasePreprocessor:
        """获取或创建预处理器"""
        key = f"{mode.value}_{input_size[0]}x{input_size[1]}"
        
        if key not in self.preprocessors:
            self.preprocessors[key] = PlantDiseasePreprocessor(
                input_size=input_size,
                mode=mode
            )
        
        return self.preprocessors[key]
    
    def process_dataset_sample(self, 
                             image_paths: List[str],
                             mode: PreprocessingMode = PreprocessingMode.TRAINING,
                             sample_size: int = 100) -> Dict[str, Any]:
        """处理数据集样本以评估预处理效果"""
        preprocessor = self.get_preprocessor(mode)
        
        # 随机采样
        import random
        sampled_paths = random.sample(image_paths, min(sample_size, len(image_paths)))
        
        results = {
            'total_processed': 0,
            'quality_issues': 0,
            'quality_distribution': {},
            'processing_times': [],
            'sample_results': []
        }
        
        for img_path in sampled_paths:
            try:
                import time
                start_time = time.time()
                
                processed_img, quality_info = preprocessor.preprocess_image(
                    img_path, return_quality_info=True
                )
                
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                results['total_processed'] += 1
                
                if not quality_info['is_good_quality']:
                    results['quality_issues'] += 1
                
                # 统计质量分布
                quality_score = quality_info['overall_quality']
                quality_bin = f"{int(quality_score * 10) * 10}-{int(quality_score * 10) * 10 + 10}%"
                results['quality_distribution'][quality_bin] = results['quality_distribution'].get(quality_bin, 0) + 1
                
                results['sample_results'].append({
                    'path': img_path,
                    'quality_score': quality_score,
                    'issues': quality_info['issues'],
                    'processing_time': processing_time
                })
                
            except Exception as e:
                logger.error(f"处理图像失败 {img_path}: {e}")
        
        # 计算统计信息
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['quality_issue_rate'] = results['quality_issues'] / results['total_processed']
        
        return results

# 全局管道实例
preprocessing_pipeline = PreprocessingPipeline()

def get_preprocessing_pipeline() -> PreprocessingPipeline:
    """获取全局预处理管道实例"""
    return preprocessing_pipeline

def create_plant_preprocessor(mode: PreprocessingMode = PreprocessingMode.TRAINING,
                            input_size: Tuple[int, int] = (224, 224)) -> PlantDiseasePreprocessor:
    """便捷函数：创建植物病害预处理器"""
    return PlantDiseasePreprocessor(input_size=input_size, mode=mode)

if __name__ == "__main__":
    # 测试图像预处理管道
    print("🧪 图像预处理管道测试")
    print("=" * 60)
    
    if not ALBUMENTATIONS_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试预处理器创建
        print("📋 测试预处理器创建...")
        
        # 创建不同模式的预处理器
        train_preprocessor = create_plant_preprocessor(PreprocessingMode.TRAINING)
        val_preprocessor = create_plant_preprocessor(PreprocessingMode.VALIDATION)
        inference_preprocessor = create_plant_preprocessor(PreprocessingMode.INFERENCE)
        
        print(f"✅ 预处理器创建成功:")
        print(f"   训练模式: {train_preprocessor.mode.value}")
        print(f"   验证模式: {val_preprocessor.mode.value}")
        print(f"   推理模式: {inference_preprocessor.mode.value}")
        
        # 创建测试图像
        print(f"\n🔍 测试图像处理...")
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 测试训练模式预处理
        processed_train = train_preprocessor.preprocess_image(test_image)
        print(f"✅ 训练模式处理完成:")
        print(f"   输出形状: {processed_train.shape}")
        print(f"   数据类型: {processed_train.dtype}")
        print(f"   数值范围: [{processed_train.min():.3f}, {processed_train.max():.3f}]")
        
        # 测试质量评估
        print(f"\n📊 测试图像质量评估...")
        quality_assessor = ImageQualityAssessment()
        quality_info = quality_assessor.assess_image_quality(test_image)
        
        print(f"✅ 质量评估完成:")
        print(f"   整体质量: {quality_info['overall_quality']:.2f}")
        print(f"   质量良好: {quality_info['is_good_quality']}")
        print(f"   分辨率: {quality_info['resolution']}")
        print(f"   亮度: {quality_info['brightness']:.1f}")
        print(f"   对比度: {quality_info['contrast']:.3f}")
        
        # 测试TTA
        print(f"\n🔄 测试测试时增强...")
        tta_batch = inference_preprocessor.create_test_time_augmentation(test_image)
        print(f"✅ TTA生成完成:")
        print(f"   批次形状: {tta_batch.shape}")
        print(f"   增强数量: {tta_batch.shape[0]}")
        
        # 测试管道管理器
        print(f"\n🔧 测试管道管理器...")
        pipeline = get_preprocessing_pipeline()
        
        # 获取预处理器
        train_proc = pipeline.get_preprocessor(PreprocessingMode.TRAINING)
        val_proc = pipeline.get_preprocessor(PreprocessingMode.VALIDATION)
        
        print(f"✅ 管道管理器测试完成:")
        print(f"   缓存的预处理器数量: {len(pipeline.preprocessors)}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 图像预处理管道测试完成")