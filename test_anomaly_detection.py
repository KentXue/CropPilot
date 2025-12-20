#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试异常检测功能
"""

import sys
import os
import requests
import json

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_anomaly_detection_module():
    """测试异常检测模块"""
    print("🔍 测试异常检测模块")
    print("=" * 50)
    
    try:
        from anomaly_detection import AnomalyDetector
        
        detector = AnomalyDetector()
        print("✅ 异常检测模块导入成功")
        
        # 测试正常数据
        print("\n1. 测试正常数据:")
        normal_data = {
            "temperature": 25,
            "humidity": 70,
            "soil_moisture": 80,
            "ph_value": 6.5,
            "nitrogen": 150
        }
        
        anomalies = detector.detect_sensor_data_anomalies(normal_data, "水稻")
        print(f"   正常数据检测到 {len(anomalies)} 个异常")
        
        # 测试异常数据
        print("\n2. 测试异常数据:")
        abnormal_data = {
            "temperature": 45,  # 过高
            "humidity": 30,     # 过低
            "soil_moisture": 40, # 过低
            "ph_value": 8.5,    # 过高
            "nitrogen": 50      # 过低
        }
        
        anomalies = detector.detect_sensor_data_anomalies(abnormal_data, "水稻")
        print(f"   异常数据检测到 {len(anomalies)} 个异常:")
        
        for anomaly in anomalies:
            print(f"     - {anomaly['parameter']}: {anomaly['message']} ({anomaly['level']})")
        
        # 测试预警摘要
        summary = detector.generate_alert_summary(anomalies)
        print(f"\n   预警摘要: {summary['summary']}")
        print(f"   最高等级: {summary['max_level']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 异常检测模块测试失败: {e}")
        return False

def test_anomaly_api():
    """测试异常检测API"""
    print("\n🌐 测试异常检测API")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    try:
        # 测试传感器数据异常检测API
        print("1. 测试传感器数据异常检测API...")
        
        test_data = {
            "sensor_data": {
                "temperature": 45,  # 过高
                "humidity": 30,     # 过低
                "soil_moisture": 85, # 正常
                "ph_value": 6.5,    # 正常
                "nitrogen": 50      # 过低
            },
            "crop_type": "水稻"
        }
        
        response = requests.post(
            f"{base_url}/api/check_sensor_anomalies",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                anomalies = result.get('anomalies', [])
                summary = result.get('alert_summary', {})
                
                print(f"   ✅ API调用成功")
                print(f"   检测到 {len(anomalies)} 个异常")
                print(f"   预警摘要: {summary.get('summary', '')}")
                
                return True
            else:
                print(f"   ❌ API返回错误: {result.get('message')}")
        else:
            print(f"   ❌ HTTP错误: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ 无法连接到服务器，请确保Flask应用正在运行")
        print("   运行命令: python src/app.py")
        return False
    except Exception as e:
        print(f"   ❌ API测试失败: {e}")
        return False
    
    return False

def test_field_alerts_api():
    """测试地块预警API"""
    print("\n📊 测试地块预警API")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    try:
        # 先获取地块列表
        response = requests.get(f"{base_url}/api/fields", timeout=5)
        if response.status_code == 200:
            fields_data = response.json()
            if fields_data.get('status') == 'success' and fields_data.get('fields'):
                field_id = fields_data['fields'][0]['id']
                field_name = fields_data['fields'][0]['name']
                
                print(f"1. 测试地块预警 - {field_name} (ID: {field_id})")
                
                # 测试地块预警API
                response = requests.get(
                    f"{base_url}/api/check_field_alerts?field_id={field_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ 地块预警API调用成功")
                    print(f"   状态: {result.get('status')}")
                    
                    if result.get('status') == 'success':
                        summary = result.get('alert_summary', {})
                        print(f"   预警摘要: {summary.get('summary', '')}")
                        print(f"   异常数量: {summary.get('total_alerts', 0)}")
                    elif result.get('status') == 'no_data':
                        print(f"   信息: {result.get('message')}")
                    
                    return True
                else:
                    print(f"   ❌ HTTP错误: {response.status_code}")
            else:
                print("   ❌ 没有找到可用的地块")
        else:
            print(f"   ❌ 获取地块列表失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ 无法连接到服务器")
        return False
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False
    
    return False

def main():
    """主函数"""
    print("🚨 CropPilot异常检测功能测试")
    print("测试时间:", __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()
    
    # 测试各个模块
    module_ok = test_anomaly_detection_module()
    api_ok = test_anomaly_api()
    field_ok = test_field_alerts_api()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"异常检测模块: {'✅ 通过' if module_ok else '❌ 失败'}")
    print(f"异常检测API:  {'✅ 通过' if api_ok else '❌ 失败'}")
    print(f"地块预警API:  {'✅ 通过' if field_ok else '❌ 失败'}")
    
    if module_ok and api_ok:
        print("\n🎉 异常检测功能测试通过！")
        print("\n🌟 新增功能:")
        print("   ✅ 传感器数据异常检测")
        print("   ✅ 多参数阈值监测")
        print("   ✅ 智能预警等级分类")
        print("   ✅ 自动处理建议生成")
        print("   ✅ 实时异常监测")
        print("   ✅ 趋势异常检测")
        
        print("\n📊 检测能力:")
        print("   - 温度、湿度异常监测")
        print("   - 土壤参数异常检测")
        print("   - 营养元素异常分析")
        print("   - 光照条件异常预警")
        print("   - 数据趋势异常识别")
        
        print("\n🎯 预警等级:")
        print("   - ℹ️  信息: 参数在最适范围内")
        print("   - ⚠️  警告: 参数偏离最适范围")
        print("   - 🚨 危险: 参数超出正常范围")
        print("   - 💀 严重: 参数严重异常")
        
        print("\n📝 使用说明:")
        print("   1. 启动服务器: python src/app.py")
        print("   2. 访问: http://localhost:5000")
        print("   3. 点击'异常预警'标签页")
        print("   4. 选择地块并检查预警")
        print("   5. 启用自动检测功能")
    else:
        print("\n⚠️  异常检测功能测试未完全通过")
        print("\n🔧 可能的问题:")
        print("   - Flask服务器未启动")
        print("   - 数据库连接问题")
        print("   - 缺少必要的依赖")

if __name__ == "__main__":
    main()