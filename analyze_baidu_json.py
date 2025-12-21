#!/usr/bin/env python3
"""
分析百度AI Studio数据集JSON文件结构
"""

import json
import os
from collections import Counter

# JSON文件路径
train_json = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\1.图像数据（病虫害识别核心）\ai_challenger_pdr2018\ai_challenger_pdr2018_trainingset_20181023\AgriculturalDisease_trainingset\AgriculturalDisease_train_annotations.json"
val_json = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\1.图像数据（病虫害识别核心）\ai_challenger_pdr2018\ai_challenger_pdr2018_validationset_20181023\AgriculturalDisease_validationset\AgriculturalDisease_validation_annotations.json"

print("🔍 分析百度AI Studio JSON文件结构")
print("=" * 50)

# 分析训练集
if os.path.exists(train_json):
    print("📋 训练集标注文件:")
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"   总样本数: {len(train_data)}")
    print(f"   数据类型: {type(train_data)}")
    
    if train_data:
        print(f"   第一个样本: {train_data[0]}")
        print(f"   样本字段: {list(train_data[0].keys())}")
        
        # 统计disease_class分布
        class_counter = Counter(item['disease_class'] for item in train_data)
        print(f"   病害类别数: {len(class_counter)}")
        print(f"   类别分布: {dict(class_counter)}")

# 分析验证集
if os.path.exists(val_json):
    print("\n📋 验证集标注文件:")
    with open(val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"   总样本数: {len(val_data)}")
    print(f"   数据类型: {type(val_data)}")
    
    if val_data:
        print(f"   第一个样本: {val_data[0]}")
        
        # 统计disease_class分布
        class_counter = Counter(item['disease_class'] for item in val_data)
        print(f"   病害类别数: {len(class_counter)}")
        print(f"   类别分布: {dict(class_counter)}")

# 查找类别映射文件
base_path = r"C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\1.图像数据（病虫害识别核心）\ai_challenger_pdr2018"
print(f"\n🔍 查找类别映射文件...")

for root, dirs, files in os.walk(base_path):
    for file in files:
        if any(keyword in file.lower() for keyword in ['class', 'label', 'category', 'disease']):
            if file.endswith(('.txt', '.json', '.csv')):
                file_path = os.path.join(root, file)
                print(f"   找到: {file_path}")
                
                # 尝试读取内容
                try:
                    if file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()[:500]  # 只读前500字符
                        print(f"     内容预览: {content}")
                    elif file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        print(f"     JSON结构: {type(data)}, 长度: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                except Exception as e:
                    print(f"     读取失败: {e}")

print("\n✅ 分析完成")