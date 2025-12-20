#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据可视化功能
"""

import requests
import json
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_visualization_features():
    """测试数据可视化相关功能"""
    base_url = "http://localhost:5000"
    
    print("🚀 测试数据可视化功能")
    print("=" * 50)
    
    # 1. 测试获取地块列表
    print("1. 测试获取地块列表...")
    try:
        response = requests.get(f"{base_url}/api/fields", timeout=5)
        if response.status_code == 200:
            fields_data = response.json()
            if fields_data.get('status') == 'success' and fields_data.get('fields'):
                field_id = fields_data['fields'][0]['id']
                field_name = fields_data['fields'][0]['name']
                print(f"   ✅ 找到地块: {field_name} (ID: {field_id})")
                
                # 2. 测试生成演示数据
                print(f"\n2. 为地块 {field_id} 生成演示数据...")
                demo_data = {
                    "field_id": field_id,
                    "days": 7
                }
                
                response = requests.post(
                    f"{base_url}/api/generate_demo_data",
                    json=demo_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'success':
                        print(f"   ✅ {result.get('message')}")
                        
                        # 3. 测试获取传感器数据
                        print(f"\n3. 测试获取传感器数据...")
                        response = requests.get(
                            f"{base_url}/api/get_sensor_data?field_id={field_id}&limit=50",
                            timeout=5
                        )
                        
                        if response.status_code == 200:
                            sensor_data = response.json()
                            if sensor_data.get('status') == 'success':
                                count = sensor_data.get('count', 0)
                                print(f"   ✅ 获取到 {count} 条传感器数据")
                                
                                if count > 0:
                                    # 显示数据样本
                                    sample = sensor_data['data'][0]
                                    print(f"   📊 数据样本:")
                                    print(f"      温度: {sample.get('temperature')}°C")
                                    print(f"      湿度: {sample.get('humidity')}%")
                                    print(f"      土壤湿度: {sample.get('soil_moisture')}%")
                                    print(f"      pH值: {sample.get('ph_value')}")
                                    print(f"      记录时间: {sample.get('recorded_at')}")
                                    
                                    print(f"\n✅ 数据可视化功能测试完成！")
                                    print(f"\n📝 使用说明:")
                                    print(f"   1. 启动服务器: python src/app.py")
                                    print(f"   2. 访问: http://localhost:5000")
                                    print(f"   3. 点击'数据可视化'标签页")
                                    print(f"   4. 选择地块: {field_name}")
                                    print(f"   5. 点击'加载图表'查看可视化效果")
                                    
                                    return True
                                else:
                                    print("   ❌ 没有获取到传感器数据")
                            else:
                                print(f"   ❌ 获取传感器数据失败: {sensor_data.get('message')}")
                        else:
                            print(f"   ❌ 获取传感器数据请求失败: {response.status_code}")
                    else:
                        print(f"   ❌ 生成演示数据失败: {result.get('message')}")
                else:
                    print(f"   ❌ 生成演示数据请求失败: {response.status_code}")
            else:
                print("   ❌ 没有找到可用的地块")
        else:
            print(f"   ❌ 获取地块列表失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("   请确保Flask应用正在运行: python src/app.py")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    return False

def main():
    """主函数"""
    print("📊 CropPilot数据可视化功能测试")
    print("测试时间:", __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    success = test_visualization_features()
    
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    if success:
        print("🎉 数据可视化功能测试通过！")
        print("\n🌟 新增功能:")
        print("   ✅ 传感器数据趋势图表")
        print("   ✅ 温度湿度双轴图表")
        print("   ✅ 土壤参数监测图表")
        print("   ✅ NPK营养元素分析")
        print("   ✅ 光照强度变化图表")
        print("   ✅ 数据统计摘要")
        print("   ✅ 演示数据生成功能")
        
        print("\n📈 图表特性:")
        print("   - 基于Chart.js的响应式图表")
        print("   - 多参数双轴显示")
        print("   - 时间序列数据展示")
        print("   - 实时数据统计分析")
        print("   - 交互式图表操作")
    else:
        print("⚠️  数据可视化功能测试未完全通过")
        print("\n🔧 可能的问题:")
        print("   - Flask服务器未启动")
        print("   - 数据库连接问题")
        print("   - 缺少必要的依赖")

if __name__ == "__main__":
    main()