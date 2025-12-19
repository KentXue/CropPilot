# 图像识别模块
# 用于作物病虫害识别（简化版本）

import os
import sys
from typing import Dict, Any, Optional
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from PIL import Image
    import numpy as np
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"图像处理依赖未安装: {e}")
    print("请运行: pip install Pillow numpy")
    IMAGE_PROCESSING_AVAILABLE = False

class SimpleImageClassifier:
    """简化的图像分类器（演示版本）"""
    
    def __init__(self):
        self.available = IMAGE_PROCESSING_AVAILABLE
        
        # 模拟的病害分类规则（基于图像特征的简单判断）
        self.disease_patterns = {
            "水稻纹枯病": {
                "description": "叶片出现椭圆形或不规则形病斑，边缘褐色",
                "treatment": "喷施三唑酮或丙环唑杀菌剂，改善田间通风"
            },
            "水稻稻瘟病": {
                "description": "叶片出现梭形病斑，中央灰白色，边缘褐色",
                "treatment": "喷施稻瘟灵或三环唑，加强水肥管理"
            },
            "玉米大斑病": {
                "description": "叶片出现长椭圆形病斑，灰褐色",
                "treatment": "喷施代森锰锌或百菌清，注意田间排水"
            },
            "玉米小斑病": {
                "description": "叶片出现小型椭圆形病斑，黄褐色",
                "treatment": "喷施多菌灵或甲基托布津，清除病残体"
            },
            "健康状态": {
                "description": "作物生长正常，无明显病害症状",
                "treatment": "继续保持良好的田间管理，定期观察"
            }
        }
    
    def analyze_image(self, image_path: str, crop_type: str = "") -> Dict[str, Any]:
        """分析图像并返回识别结果"""
        if not self.available:
            return {
                "status": "error",
                "message": "图像处理功能不可用，请安装相关依赖"
            }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return {
                    "status": "error",
                    "message": "图像文件不存在"
                }
            
            # 打开并分析图像
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 获取图像基本信息
                width, height = img.size
                img_array = np.array(img)
                
                # 简化的特征分析（实际项目中应使用深度学习模型）
                result = self._simple_feature_analysis(img_array, crop_type)
                
                return {
                    "status": "success",
                    "image_info": {
                        "width": width,
                        "height": height,
                        "format": img.format,
                        "mode": img.mode
                    },
                    "analysis_result": result
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "message": f"图像分析失败: {str(e)}"
            }
    
    def _simple_feature_analysis(self, img_array: np.ndarray, crop_type: str) -> Dict[str, Any]:
        """简化的特征分析（模拟深度学习模型的输出）"""
        
        # 计算图像的基本统计特征
        mean_rgb = np.mean(img_array, axis=(0, 1))
        std_rgb = np.std(img_array, axis=(0, 1))
        
        # 基于简单规则的病害判断（实际应用中需要训练好的模型）
        green_ratio = mean_rgb[1] / (mean_rgb[0] + mean_rgb[1] + mean_rgb[2])
        color_variance = np.mean(std_rgb)
        
        # 模拟识别逻辑
        if green_ratio > 0.4 and color_variance < 30:
            # 绿色占比高，颜色变化小 -> 健康
            disease = "健康状态"
            confidence = 0.85
        elif crop_type == "水稻":
            if color_variance > 50:
                disease = "水稻纹枯病"
                confidence = 0.72
            else:
                disease = "水稻稻瘟病"
                confidence = 0.68
        elif crop_type == "玉米":
            if mean_rgb[1] < 100:  # 绿色值较低
                disease = "玉米大斑病"
                confidence = 0.75
            else:
                disease = "玉米小斑病"
                confidence = 0.70
        else:
            # 未知作物类型，给出通用建议
            if green_ratio < 0.3:
                disease = "疑似病害"
                confidence = 0.60
            else:
                disease = "健康状态"
                confidence = 0.65
        
        disease_info = self.disease_patterns.get(disease, {
            "description": "未知病害类型",
            "treatment": "建议咨询当地农技专家"
        })
        
        return {
            "disease_name": disease,
            "confidence": confidence,
            "description": disease_info["description"],
            "treatment_advice": disease_info["treatment"],
            "image_features": {
                "green_ratio": float(green_ratio),
                "color_variance": float(color_variance),
                "mean_rgb": mean_rgb.tolist()
            }
        }
    
    def get_disease_info(self, disease_name: str) -> Dict[str, str]:
        """获取特定病害的详细信息"""
        return self.disease_patterns.get(disease_name, {
            "description": "未知病害",
            "treatment": "请咨询专业人员"
        })

# 全局实例
image_classifier = None

def get_image_classifier():
    """获取图像分类器实例（单例模式）"""
    global image_classifier
    if image_classifier is None:
        image_classifier = SimpleImageClassifier()
    return image_classifier

def analyze_crop_image(image_path: str, crop_type: str = "") -> Dict[str, Any]:
    """便捷的图像分析函数"""
    classifier = get_image_classifier()
    return classifier.analyze_image(image_path, crop_type)

if __name__ == "__main__":
    # 测试图像识别功能
    print("测试图像识别模块...")
    
    classifier = SimpleImageClassifier()
    if classifier.available:
        print("图像识别模块可用")
        
        # 显示支持的病害类型
        print("\n支持识别的病害类型:")
        for disease, info in classifier.disease_patterns.items():
            print(f"- {disease}: {info['description']}")
    else:
        print("图像识别模块不可用，请安装相关依赖")