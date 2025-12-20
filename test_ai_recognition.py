#!/usr/bin/env python3
"""
测试AI图像识别功能

这个脚本用于测试CropPilot的AI图像识别功能
"""

import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ai_module():
    """测试AI图像识别模块"""
    print("🧪 测试CropPilot AI图像识别模块")
    print("=" * 50)
    
    try:
        from image_recognition import get_plant_classifier, analyze_crop_image
        
        # 获取分类器实例
        classifier = get_plant_classifier()
        
        print(f"📊 模块状态:")
        print(f"   - AI可用: {classifier.available}")
        
        if classifier.available:
            print(f"   - 设备: {classifier.device}")
            print(f"   - 模型类型: ResNet18")
            print(f"   - 支持类别: {len(classifier.class_names)}种")
            
            # 显示支持的病害类型
            print(f"\n🔍 支持识别的病害类型:")
            for i, (english, chinese) in enumerate(list(classifier.chinese_names.items())[:10]):
                print(f"   {i+1:2d}. {chinese}")
                print(f"       ({english})")
            
            if len(classifier.chinese_names) > 10:
                print(f"   ... 还有{len(classifier.chinese_names) - 10}种病害")
            
            print(f"\n💊 治疗建议示例:")
            for disease, advice in list(classifier.treatment_advice.items())[:3]:
                chinese_name = classifier.chinese_names.get(disease, disease)
                print(f"   - {chinese_name}: {advice}")
            
        else:
            print("   - 使用基础规则识别")
            print("   - 提示: 安装PyTorch获得完整AI功能")
            print("     pip install torch torchvision")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_image_analysis():
    """测试图像分析功能（使用示例图片）"""
    print(f"\n🖼️  测试图像分析功能")
    print("-" * 30)
    
    try:
        from image_recognition import analyze_crop_image
        
        # 创建一个测试图片（如果不存在）
        test_image_path = create_test_image()
        
        if test_image_path and os.path.exists(test_image_path):
            print(f"📁 测试图片: {test_image_path}")
            
            # 测试不同作物类型
            crop_types = ["玉米", "水稻", ""]
            
            for crop_type in crop_types:
                print(f"\n🌾 测试作物类型: {crop_type or '未指定'}")
                
                result = analyze_crop_image(test_image_path, crop_type)
                
                if result.get('status') == 'success':
                    method = result.get('method', 'unknown')
                    print(f"   ✅ 识别成功 (方法: {method})")
                    
                    analysis = result.get('analysis_result', {})
                    primary = analysis.get('primary_result', analysis)
                    
                    disease = primary.get('disease_name', '未知')
                    confidence = primary.get('confidence', 0)
                    treatment = primary.get('treatment_advice', '无建议')
                    
                    print(f"   🔍 识别结果: {disease}")
                    print(f"   📊 置信度: {confidence:.2%}")
                    print(f"   💊 建议: {treatment}")
                    
                    # 显示备选结果（如果有）
                    alternatives = analysis.get('alternative_results', [])
                    if alternatives:
                        print(f"   📋 备选结果:")
                        for i, alt in enumerate(alternatives[:2]):
                            alt_disease = alt.get('disease_name', '未知')
                            alt_conf = alt.get('confidence', 0)
                            print(f"      {i+1}. {alt_disease} ({alt_conf:.1%})")
                else:
                    print(f"   ❌ 识别失败: {result.get('message', '未知错误')}")
        else:
            print("⚠️  无测试图片，跳过图像分析测试")
            
    except Exception as e:
        print(f"❌ 图像分析测试失败: {e}")

def create_test_image():
    """创建一个简单的测试图片"""
    try:
        from PIL import Image
        import numpy as np
        
        # 创建一个简单的绿色图片（模拟健康植物）
        width, height = 224, 224
        
        # 创建绿色背景
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        img_array[:, :, 1] = 120  # 绿色通道
        img_array[:, :, 0] = 60   # 红色通道
        img_array[:, :, 2] = 40   # 蓝色通道
        
        # 添加一些随机变化（模拟叶片纹理）
        noise = np.random.randint(-20, 20, (height, width, 3))
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # 保存图片
        img = Image.fromarray(img_array)
        test_path = "test_plant_image.jpg"
        img.save(test_path)
        
        print(f"✅ 创建测试图片: {test_path}")
        return test_path
        
    except ImportError:
        print("⚠️  PIL未安装，无法创建测试图片")
        return None
    except Exception as e:
        print(f"❌ 创建测试图片失败: {e}")
        return None

def test_api_endpoints():
    """测试API端点（需要Flask应用运行）"""
    print(f"\n🌐 API端点测试")
    print("-" * 30)
    
    try:
        import requests
        
        base_url = "http://localhost:5000"
        
        # 测试支持的病害列表API
        print("📡 测试 GET /api/get_supported_diseases")
        try:
            response = requests.get(f"{base_url}/api/get_supported_diseases", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    diseases = data.get('supported_diseases', [])
                    ai_available = data.get('ai_available', False)
                    print(f"   ✅ 成功 - AI可用: {ai_available}, 支持病害: {len(diseases)}种")
                else:
                    print(f"   ❌ API返回错误: {data}")
            else:
                print(f"   ❌ HTTP错误: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("   ⚠️  连接失败 - 请确保Flask应用正在运行")
            print("      启动命令: python src/app.py")
        except Exception as e:
            print(f"   ❌ 请求失败: {e}")
            
    except ImportError:
        print("⚠️  requests未安装，跳过API测试")
        print("   安装: pip install requests")

def main():
    """主测试函数"""
    print("🌾 CropPilot AI图像识别功能测试")
    print("=" * 60)
    
    # 测试AI模块
    if test_ai_module():
        # 测试图像分析
        test_image_analysis()
        
        # 测试API端点
        test_api_endpoints()
        
        print(f"\n" + "=" * 60)
        print("🎉 测试完成!")
        print("\n💡 使用建议:")
        print("   1. 启动应用: python src/app.py")
        print("   2. 访问Web界面: http://localhost:5000")
        print("   3. 上传真实的植物图片进行识别")
        print("   4. 查看识别结果和治疗建议")
        
        print("\n📚 相关文件:")
        print("   - src/image_recognition.py (AI识别模块)")
        print("   - download_model.py (模型下载脚本)")
        print("   - requirements.txt (依赖列表)")
        
    else:
        print("\n❌ AI模块测试失败")
        print("💡 解决方案:")
        print("   1. 安装依赖: pip install torch torchvision Pillow numpy")
        print("   2. 运行模型设置: python download_model.py")
        print("   3. 重新测试: python test_ai_recognition.py")

if __name__ == "__main__":
    main()