#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from smart_knowledge import SmartKnowledgeBase

def test_vector_search():
    print("测试向量搜索...")
    
    kb = SmartKnowledgeBase()
    if not kb.available:
        print("智能知识库不可用")
        return
    
    # 测试查询
    question = "叶子发黄怎么办"
    crop_type = "水稻"
    growth_stage = "分蘖期"
    
    print(f"查询: {question} (作物: {crop_type}, 阶段: {growth_stage})")
    
    results = kb.query(question, crop_type, growth_stage)
    print(f"返回 {len(results)} 条结果:")
    
    for i, result in enumerate(results, 1):
        relevance = result.get('relevance_score', 0)
        distance = result.get('distance', 0)
        content = result.get('content', '')
        source = result.get('source', '')
        
        print(f"\n{i}. 距离: {distance:.3f}, 相关度: {relevance:.3f}")
        print(f"   来源: {source}")
        print(f"   内容: {content[:100]}...")
    
    # 测试格式化输出
    print("\n" + "="*50)
    print("格式化建议:")
    advice = kb.format_advice(results, question)
    print(advice)

if __name__ == "__main__":
    test_vector_search()