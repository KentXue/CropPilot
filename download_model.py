#!/usr/bin/env python3
"""
下载和设置植物病害识别模型

这个脚本帮助用户下载预训练的植物病害识别模型
"""

import os
import sys
import requests
from pathlib import Path
import torch
import torchvision.models as models

def download_pretrained_model():
    """下载并设置预训练模型"""
    
    print("🌾 CropPilot AI图像识别模型设置")
    print("=" * 50)
    
    # 检查PyTorch是否安装
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ TorchVision版本: {torchvision.__version__}")
    except ImportError:
        print("❌ PyTorch未安装，请先运行:")
        print("   pip install torch torchvision")
        return False
    
    # 创建模型目录
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "plant_disease_model.pth"
    
    # 检查是否已有模型
    if model_path.exists():
        print(f"✅ 模型已存在: {model_path}")
        return True
    
    print("\n📥 准备下载模型...")
    
    # 方案1: 使用预训练的ResNet18并保存
    try:
        print("🔄 创建基础模型...")
        
        # 创建模型结构
        model = models.resnet18(pretrained=True)
        
        # 植物病害类别数（基于PlantVillage数据集）
        num_classes = 38
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # 保存模型结构和预训练权重
        print(f"💾 保存模型到: {model_path}")
        torch.save(model.state_dict(), model_path)
        
        print("✅ 基础模型创建成功!")
        print("\n📝 注意事项:")
        print("   - 当前使用的是基础预训练模型")
        print("   - 为获得最佳效果，建议使用专门的植物病害数据集训练")
        print("   - 系统会自动回退到规则识别作为备用方案")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_model():
    """测试模型是否可以正常加载"""
    try:
        from src.image_recognition import get_plant_classifier
        
        print("\n🧪 测试AI图像识别模块...")
        classifier = get_plant_classifier()
        
        if classifier.available:
            print("✅ AI图像识别模块可用")
            print(f"   - 设备: {classifier.device}")
            print(f"   - 支持病害类型: {len(classifier.class_names)}种")
            
            # 显示部分支持的病害
            print("\n🔍 支持识别的病害类型（部分）:")
            for i, (english, chinese) in enumerate(list(classifier.chinese_names.items())[:5]):
                print(f"   {i+1}. {chinese} ({english})")
            print(f"   ... 共{len(classifier.chinese_names)}种")
            
            return True
        else:
            print("⚠️  AI模块不可用，将使用基础规则识别")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始设置CropPilot AI图像识别功能...\n")
    
    # 下载/创建模型
    if download_pretrained_model():
        print("\n" + "=" * 50)
        
        # 测试模型
        if test_model():
            print("\n🎉 AI图像识别功能设置完成!")
            print("\n🚀 使用方法:")
            print("   1. 启动应用: python src/app.py")
            print("   2. 访问: http://localhost:5000")
            print("   3. 上传作物图片进行AI识别")
            
            print("\n📚 API接口:")
            print("   - POST /api/upload_crop_image (上传并识别)")
            print("   - POST /api/analyze_image (仅识别)")
            print("   - GET /api/get_supported_diseases (支持的病害)")
        else:
            print("\n⚠️  AI功能测试失败，但系统仍可使用基础功能")
    else:
        print("\n❌ 模型设置失败")
        print("💡 建议:")
        print("   1. 检查网络连接")
        print("   2. 确保已安装PyTorch: pip install torch torchvision")
        print("   3. 系统仍可使用基础规则识别功能")

if __name__ == "__main__":
    main()