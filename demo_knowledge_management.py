#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示知识库管理功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_loader import KnowledgeLoader

def demo_knowledge_management():
    """演示知识库管理"""
    print("🌾 农业知识库管理演示")
    print("=" * 60)
    
    loader = KnowledgeLoader()
    
    # 1. 查看现有知识
    print("1️⃣ 查看现有知识库:")
    docs = loader.load_from_json()
    print(f"   📊 共有 {len(docs)} 条知识")
    
    for i, doc in enumerate(docs[:3], 1):  # 只显示前3条
        print(f"   {i}. [{doc['crop']}-{doc['stage']}] {doc['content'][:40]}...")
    
    if len(docs) > 3:
        print(f"   ... 还有 {len(docs) - 3} 条知识")
    
    # 2. 添加新知识
    print(f"\n2️⃣ 添加新知识:")
    new_doc = loader.add_document(
        content="小麦播种期要选择适宜的播种时间，一般在10月中下旬。播种深度3-4cm，行距15-20cm。播种后要及时镇压保墒。",
        source="小麦栽培技术规程",
        crop="小麦",
        stage="播种期",
        keywords=["小麦", "播种", "镇压", "保墒"],
        priority=1
    )
    print(f"   ✅ 新增知识ID: {new_doc['id']}")
    
    # 3. 保存到文件
    print(f"\n3️⃣ 保存知识库:")
    docs.append(new_doc)
    success = loader.save_to_json(docs)
    if success:
        print("   ✅ 知识库已保存到 data/agriculture_knowledge.json")
    
    # 4. 重新加载验证
    print(f"\n4️⃣ 验证保存结果:")
    updated_docs = loader.load_from_json()
    print(f"   📊 更新后共有 {len(updated_docs)} 条知识")
    
    # 查找新添加的知识
    new_knowledge = [doc for doc in updated_docs if doc['crop'] == '小麦']
    if new_knowledge:
        print(f"   🆕 找到小麦相关知识: {len(new_knowledge)} 条")
        print(f"      内容: {new_knowledge[0]['content'][:50]}...")
    
    print(f"\n✅ 演示完成！")
    print(f"\n💡 使用方法:")
    print(f"   - 直接编辑 data/agriculture_knowledge.json 文件")
    print(f"   - 使用 python manage_knowledge.py 进行交互式管理")
    print(f"   - 通过代码调用 KnowledgeLoader 类进行程序化管理")

if __name__ == "__main__":
    demo_knowledge_management()