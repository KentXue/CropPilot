#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CropPilot智能功能测试脚本
测试智能知识库和图像识别功能
"""

import os
import sys
import requests
import json

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_smart_knowledge():
    """测试智能知识库功能"""
    print("=" * 50)
    print("测试智能知识库功能")
    print("=" * 50)
    
    try:
        from src.smart_knowledge import SmartKnowledgeBase, smart_query
        
        # 创建知识库实例
        kb = SmartKnowledgeBase()
        
        if not kb.available:
            print("❌ 智能知识库不可用，请安装依赖：")
            print("   pip install chromadb sentence-transformers")
            return False
        
        print("✅ 智能知识库初始化成功")
        
        # 测试查询
        test_queries = [
            ("叶子发黄怎么办", "水稻", "分蘖期"),
            ("如何施肥", "玉米", "拔节期"), 
            ("病虫害防治", "", ""),
            ("高温干旱应对", "", ""),
            ("土壤pH值调节", "", "")
        ]
        
        print(f"\n📋 测试 {len(test_queries)} 个查询...")
        
        for i, (question, crop, stage) in enumerate(test_queries, 1):
            print(f"\n{i}. 查询: {question}")
            if crop or stage:
                print(f"   作物: {crop}, 阶段: {stage}")
            
            result = smart_query(question, crop, stage)
            print(f"   结果: {result[:100]}...")
            
        print("\n✅ 智能知识库测试完成")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_image_recognition():
    """测试图像识别功能"""
    print("\n" + "=" * 50)
    print("测试图像识别功能")
    print("=" * 50)
    
    try:
        from src.image_recognition import SimpleImageClassifier, analyze_crop_image
        
        # 创建分类器实例
        classifier = SimpleImageClassifier()
        
        if not classifier.available:
            print("❌ 图像识别不可用，请安装依赖：")
            print("   pip install Pillow numpy")
            return False
        
        print("✅ 图像识别模块初始化成功")
        
        # 显示支持的病害类型
        print(f"\n📋 支持识别的病害类型 ({len(classifier.disease_patterns)} 种):")
        for disease, info in classifier.disease_patterns.items():
            print(f"   - {disease}: {info['description']}")
        
        print("\n✅ 图像识别测试完成")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_api_endpoints():
    """测试API接口"""
    print("\n" + "=" * 50)
    print("测试API接口")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # 测试智能建议API
    print("📡 测试智能建议API...")
    
    test_data = {
        "question": "水稻叶子发黄怎么办",
        "crop_type": "水稻",
        "growth_stage": "分蘖期"
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/smart_advice",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                print("✅ 智能建议API测试成功")
                print(f"   问题: {result.get('question')}")
                print(f"   建议: {result.get('advice', '')[:100]}...")
            else:
                print(f"❌ API返回错误: {result.get('message')}")
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Flask应用正在运行")
        print("   运行命令: python src/app.py")
    except Exception as e:
        print(f"❌ API测试失败: {e}")

def main():
    """主测试函数"""
    print("🚀 CropPilot智能功能测试")
    print("测试时间:", __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 测试各个模块
    kb_ok = test_smart_knowledge()
    img_ok = test_image_recognition()
    
    # 测试API（需要服务器运行）
    test_api_endpoints()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"智能知识库: {'✅ 通过' if kb_ok else '❌ 失败'}")
    print(f"图像识别:   {'✅ 通过' if img_ok else '❌ 失败'}")
    
    if kb_ok and img_ok:
        print("\n🎉 所有智能功能测试通过！")
        print("\n📝 下一步建议:")
        print("   1. 运行 'python src/app.py' 启动服务器")
        print("   2. 访问 http://localhost:5000 测试Web界面")
        print("   3. 尝试智能咨询功能")
    else:
        print("\n⚠️  部分功能测试失败，请检查依赖安装")
        print("\n🔧 安装命令:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()