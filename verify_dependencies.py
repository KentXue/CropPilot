#!/usr/bin/env python3
"""
验证AI图像识别依赖安装
"""

def check_dependencies():
    """检查所有必要依赖是否正确安装"""
    print("🔍 验证AI图像识别依赖安装...")
    print("=" * 50)
    
    dependencies = [
        ("torch", "PyTorch深度学习框架"),
        ("torchvision", "PyTorch视觉库"),
        ("efficientnet_pytorch", "EfficientNet模型"),
        ("albumentations", "数据增强库"),
        ("onnx", "模型优化工具"),
        ("cv2", "OpenCV图像处理"),
        ("PIL", "Python图像库"),
        ("numpy", "数值计算库")
    ]
    
    success_count = 0
    
    for module_name, description in dependencies:
        try:
            if module_name == "cv2":
                import cv2
                version = cv2.__version__
            elif module_name == "PIL":
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else "已安装"
            elif module_name == "efficientnet_pytorch":
                from efficientnet_pytorch import EfficientNet
                version = "已安装"
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', '已安装')
            
            print(f"✅ {description}: {version}")
            success_count += 1
            
        except ImportError as e:
            print(f"❌ {description}: 未安装 - {e}")
        except Exception as e:
            print(f"⚠️  {description}: 安装异常 - {e}")
    
    print(f"\n📊 依赖检查结果: {success_count}/{len(dependencies)} 成功")
    
    if success_count == len(dependencies):
        print("🎉 所有依赖安装成功！可以开始AI模型开发")
        return True
    else:
        print("⚠️  部分依赖安装失败，请检查安装过程")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print(f"\n🧪 测试基本功能...")
    print("-" * 30)
    
    try:
        # 测试PyTorch
        import torch
        print(f"✅ PyTorch: 设备支持 - {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # 测试EfficientNet
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_name('efficientnet-b0')
        print(f"✅ EfficientNet: 模型创建成功")
        
        # 测试Albumentations
        import albumentations as A
        transform = A.Compose([A.Resize(224, 224)])
        print(f"✅ Albumentations: 数据增强管道创建成功")
        
        # 测试OpenCV
        import cv2
        print(f"✅ OpenCV: 版本 {cv2.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CropPilot AI图像识别依赖验证")
    print("=" * 60)
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    if deps_ok:
        # 测试功能
        func_ok = test_basic_functionality()
        
        if func_ok:
            print(f"\n🎯 验证完成: 环境准备就绪！")
            print("📋 下一步: 可以开始任务1.1 - 创建数据集管理模块")
        else:
            print(f"\n⚠️  验证完成: 依赖已安装但功能测试失败")
    else:
        print(f"\n❌ 验证失败: 请重新安装缺失的依赖")