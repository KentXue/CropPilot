# 真实AI图像识别模块
# 用于作物病虫害识别（基于深度学习模型）

import os
import sys
from typing import Dict, Any, Optional, List
import logging
import requests
from io import BytesIO

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from PIL import Image
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    import torch.nn.functional as F
    AI_AVAILABLE = True
except ImportError as e:
    print(f"AI依赖未安装: {e}")
    print("请运行: pip install torch torchvision Pillow numpy")
    AI_AVAILABLE = False

class PlantDiseaseClassifier:
    """基于深度学习的植物病害识别器"""
    
    def __init__(self):
        self.available = AI_AVAILABLE
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 植物病害类别映射（基于PlantVillage数据集）
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        # 中文病害名称映射
        self.chinese_names = {
            'Apple___Apple_scab': '苹果黑星病',
            'Apple___Black_rot': '苹果黑腐病',
            'Apple___Cedar_apple_rust': '苹果锈病',
            'Apple___healthy': '苹果健康',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': '玉米灰斑病',
            'Corn_(maize)___Common_rust_': '玉米锈病',
            'Corn_(maize)___Northern_Leaf_Blight': '玉米大斑病',
            'Corn_(maize)___healthy': '玉米健康',
            'Tomato___Bacterial_spot': '番茄细菌性斑点病',
            'Tomato___Early_blight': '番茄早疫病',
            'Tomato___Late_blight': '番茄晚疫病',
            'Tomato___Leaf_Mold': '番茄叶霉病',
            'Tomato___healthy': '番茄健康',
            'Potato___Early_blight': '马铃薯早疫病',
            'Potato___Late_blight': '马铃薯晚疫病',
            'Potato___healthy': '马铃薯健康'
        }
        
        # 治疗建议
        self.treatment_advice = {
            'Apple___Apple_scab': '喷施多菌灵或代森锰锌，清除落叶，改善通风',
            'Apple___Black_rot': '剪除病枝，喷施铜制剂杀菌剂',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': '喷施三唑类杀菌剂，注意田间排水',
            'Corn_(maize)___Common_rust_': '喷施丙环唑或戊唑醇，加强田间管理',
            'Corn_(maize)___Northern_Leaf_Blight': '喷施代森锰锌，清除病残体',
            'Tomato___Bacterial_spot': '喷施铜制剂，避免叶面湿润',
            'Tomato___Early_blight': '喷施百菌清或甲基托布津',
            'Tomato___Late_blight': '喷施霜脲氰或烯酰吗啉',
            'Potato___Early_blight': '喷施代森锰锌，控制湿度',
            'Potato___Late_blight': '喷施霜脲氰，及时排水'
        }
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化模型
        if self.available:
            self._load_model()
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 基于数据集质量分析，使用更稳定的ResNet18架构
            # EfficientNet-B4在当前数据不平衡情况下容易过拟合
            self.model = models.resnet18(pretrained=True)
            
            # 修改最后一层以适应我们的类别数
            num_classes = len(self.class_names)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
            print("使用ResNet18架构（更适合当前数据集）")
            
            # 尝试加载我们训练好的权重（如果有的话）
            model_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'plant_disease_gpu', 'best_model.pth'),
                os.path.join(os.path.dirname(__file__), '..', 'models', 'plant_disease_model.pth')
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    print(f"尝试加载训练好的模型: {model_path}")
                    try:
                        # 尝试加载完整的checkpoint
                        checkpoint = torch.load(model_path, map_location=self.device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            # 检查模型架构是否匹配
                            try:
                                self.model.load_state_dict(checkpoint['model_state_dict'])
                                model_loaded = True
                                print(f"✅ 成功加载训练好的模型")
                                break
                            except RuntimeError as e:
                                print(f"⚠️ 模型架构不匹配，跳过: {e}")
                                continue
                        else:
                            # 如果只是模型权重
                            try:
                                self.model.load_state_dict(checkpoint)
                                model_loaded = True
                                print(f"✅ 成功加载训练好的模型")
                                break
                            except RuntimeError as e:
                                print(f"⚠️ 模型架构不匹配，跳过: {e}")
                                continue
                    except Exception as e:
                        print(f"⚠️ 加载模型失败: {e}")
                        continue
            
            if not model_loaded:
                print("未找到兼容的训练模型，使用预训练的ResNet18")
                print("建议：重新训练ResNet18模型以获得更好的性能")
                # 如果没有训练好的模型，我们使用一个简化的方法
                # 在实际应用中，您需要使用植物病害数据集训练模型
                
            self.model.to(self.device)
            self.model.eval()
            
            print(f"模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.available = False
    
    def analyze_image(self, image_path: str, crop_type: str = "") -> Dict[str, Any]:
        """分析图像并返回识别结果"""
        if not self.available or self.model is None:
            return self._fallback_analysis(image_path, crop_type)
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                return {
                    "status": "error",
                    "message": "图像文件不存在"
                }
            
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 进行推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # 获取前3个最可能的结果
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                
                results = []
                for i in range(3):
                    class_idx = top3_indices[0][i].item()
                    prob = top3_prob[0][i].item()
                    class_name = self.class_names[class_idx]
                    chinese_name = self.chinese_names.get(class_name, class_name)
                    
                    results.append({
                        "disease_name": chinese_name,
                        "english_name": class_name,
                        "confidence": float(prob),
                        "treatment_advice": self.treatment_advice.get(class_name, "请咨询农业专家")
                    })
            
            return {
                "status": "success",
                "method": "deep_learning",
                "image_info": {
                    "width": image.size[0],
                    "height": image.size[1],
                    "format": image.format,
                    "mode": image.mode
                },
                "analysis_result": {
                    "primary_result": results[0],
                    "alternative_results": results[1:],
                    "model_info": {
                        "model_type": "ResNet18",
                        "device": str(self.device),
                        "num_classes": len(self.class_names)
                    }
                }
            }
            
        except Exception as e:
            print(f"AI识别失败，使用备用方法: {e}")
            return self._fallback_analysis(image_path, crop_type)
    
    def _fallback_analysis(self, image_path: str, crop_type: str) -> Dict[str, Any]:
        """备用分析方法（基于简单规则）"""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                width, height = img.size
                img_array = np.array(img)
                
                # 简化的特征分析
                mean_rgb = np.mean(img_array, axis=(0, 1))
                std_rgb = np.std(img_array, axis=(0, 1))
                green_ratio = mean_rgb[1] / (mean_rgb[0] + mean_rgb[1] + mean_rgb[2])
                color_variance = np.mean(std_rgb)
                
                # 基于规则的判断
                if green_ratio > 0.4 and color_variance < 30:
                    disease = "健康状态"
                    confidence = 0.75
                    treatment = "继续保持良好的田间管理"
                elif crop_type.lower() in ["玉米", "corn", "maize"]:
                    if color_variance > 50:
                        disease = "玉米大斑病"
                        confidence = 0.65
                        treatment = "喷施代森锰锌，清除病残体"
                    else:
                        disease = "玉米锈病"
                        confidence = 0.60
                        treatment = "喷施丙环唑，加强田间管理"
                else:
                    disease = "疑似病害"
                    confidence = 0.50
                    treatment = "建议上传更清晰的图片或咨询专家"
                
                return {
                    "status": "success",
                    "method": "rule_based",
                    "image_info": {
                        "width": width,
                        "height": height,
                        "format": img.format,
                        "mode": img.mode
                    },
                    "analysis_result": {
                        "primary_result": {
                            "disease_name": disease,
                            "confidence": confidence,
                            "treatment_advice": treatment
                        },
                        "image_features": {
                            "green_ratio": float(green_ratio),
                            "color_variance": float(color_variance)
                        }
                    }
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"图像分析失败: {str(e)}"
            }
    
    def batch_analyze(self, image_paths: List[str], crop_type: str = "") -> List[Dict[str, Any]]:
        """批量分析图像"""
        results = []
        for image_path in image_paths:
            result = self.analyze_image(image_path, crop_type)
            results.append(result)
        return results
    
    def get_supported_diseases(self) -> List[str]:
        """获取支持识别的病害列表"""
        return list(self.chinese_names.values())

# 全局实例
plant_classifier = None

def get_plant_classifier():
    """获取植物病害分类器实例（单例模式）"""
    global plant_classifier
    if plant_classifier is None:
        plant_classifier = PlantDiseaseClassifier()
    return plant_classifier

def analyze_crop_image(image_path: str, crop_type: str = "") -> Dict[str, Any]:
    """便捷的图像分析函数"""
    classifier = get_plant_classifier()
    return classifier.analyze_image(image_path, crop_type)

def download_sample_model():
    """下载示例模型（如果需要）"""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 这里可以添加从网络下载预训练模型的代码
    # 实际项目中，您需要训练或获取一个植物病害识别模型
    print("模型下载功能待实现，当前使用预训练的ResNet18")

if __name__ == "__main__":
    # 测试AI图像识别功能
    print("测试AI图像识别模块...")
    
    classifier = PlantDiseaseClassifier()
    if classifier.available:
        print("AI图像识别模块可用")
        print(f"支持的设备: {classifier.device}")
        print(f"支持识别 {len(classifier.class_names)} 种病害")
        
        # 显示部分支持的病害类型
        print("\n支持识别的病害类型（部分）:")
        for english, chinese in list(classifier.chinese_names.items())[:10]:
            print(f"- {chinese} ({english})")
            
        print("\n如需完整功能，请确保安装了PyTorch:")
        print("pip install torch torchvision")
    else:
        print("AI图像识别模块不可用，请安装相关依赖")
        print("pip install torch torchvision Pillow numpy")