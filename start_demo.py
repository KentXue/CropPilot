#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CropPilot演示启动脚本
"""

import os
import sys
import time
import subprocess
import webbrowser
from threading import Timer

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查系统依赖...")
    
    required_packages = [
        'flask', 'pymysql', 'chromadb', 
        'sentence-transformers', 'pillow', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def check_database():
    """检查数据库连接"""
    print("\n🗄️  检查数据库连接...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from database import get_connection
        
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM fields")
            result = cursor.fetchone()
            field_count = result['count'] if result else 0
            
        conn.close()
        print(f"   ✅ 数据库连接成功，找到 {field_count} 个地块")
        return True
        
    except Exception as e:
        print(f"   ❌ 数据库连接失败: {e}")
        print("   请检查:")
        print("   1. MySQL服务是否启动")
        print("   2. .env文件配置是否正确")
        print("   3. 数据库是否已初始化")
        return False

def check_smart_knowledge():
    """检查智能知识库"""
    print("\n🧠 检查智能知识库...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from smart_knowledge import SmartKnowledgeBase
        
        kb = SmartKnowledgeBase()
        if kb.available:
            doc_count = kb.collection.count()
            print(f"   ✅ 智能知识库可用，包含 {doc_count} 条知识")
            return True
        else:
            print("   ❌ 智能知识库不可用")
            return False
            
    except Exception as e:
        print(f"   ❌ 智能知识库检查失败: {e}")
        return False

def open_browser():
    """延迟打开浏览器"""
    time.sleep(3)  # 等待服务器启动
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 已自动打开浏览器")
    except:
        print("🌐 请手动访问: http://localhost:5000")

def main():
    """主函数"""
    print("🚀 CropPilot系统启动检查")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查数据库
    if not check_database():
        return
    
    # 检查智能知识库
    check_smart_knowledge()  # 不强制要求
    
    print("\n" + "=" * 50)
    print("🎉 系统检查完成，准备启动服务器...")
    print("=" * 50)
    
    # 设置自动打开浏览器
    timer = Timer(3.0, open_browser)
    timer.start()
    
    print("\n📝 使用提示:")
    print("   1. 系统启动后会自动打开浏览器")
    print("   2. 点击'数据可视化'标签页")
    print("   3. 选择地块并生成演示数据")
    print("   4. 查看图表效果")
    print("   5. 按 Ctrl+C 停止服务器")
    
    print("\n🚀 启动Flask服务器...")
    print("-" * 50)
    
    try:
        # 启动Flask应用
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run([sys.executable, 'src/app.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")

if __name__ == "__main__":
    main()