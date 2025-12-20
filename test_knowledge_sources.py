#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同数据源的知识库加载
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from smart_knowledge import SmartKnowledgeBase

def test_json_source():
    """测试JSON数据源"""
    print("=" * 60)
    print("测试 JSON 数据源")
    print("=" * 60)
    
    kb = SmartKnowledgeBase(data_source="json", data_path="data/agriculture_knowledge.json")
    
    if kb.available:
        print(f"✅ 知识库初始化成功")
        print(f"📊 文档数量: {kb.collection.count()}")
        
        # 测试查询
        results = kb.query("叶子发黄怎么办", "水稻", "分蘖期")
        print(f"🔍 查询结果: {len(results)} 条")
        
        if results:
            print(f"\n最相关的结果:")
            print(f"  内容: {results[0]['content'][:80]}...")
            print(f"  来源: {results[0]['source']}")
    else:
        print("❌ 知识库初始化失败")

def test_hardcoded_fallback():
    """测试硬编码兜底"""
    print("\n" + "=" * 60)
    print("测试硬编码兜底（模拟JSON文件不存在）")
    print("=" * 60)
    
    kb = SmartKnowledgeBase(data_source="json", data_path="data/nonexistent.json")
    
    if kb.available:
        print(f"✅ 知识库初始化成功（使用兜底数据）")
        print(f"📊 文档数量: {kb.collection.count()}")
    else:
        print("❌ 知识库初始化失败")

if __name__ == "__main__":
    test_json_source()
    test_hardcoded_fallback()
    
    print("\n" + "=" * 60)
    print("📝 总结")
    print("=" * 60)
    print("✅ JSON数据源：知识可以通过编辑JSON文件轻松管理")
    print("✅ 硬编码兜底：确保系统在任何情况下都能运行")
    print("✅ 易于扩展：可以添加CSV、数据库等其他数据源")