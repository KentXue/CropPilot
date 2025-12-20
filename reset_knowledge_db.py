#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重置知识库向量数据库
用于在更新JSON文件后重新构建向量索引
"""

import sys
import os
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from smart_knowledge import SmartKnowledgeBase

def reset_vector_database():
    """重置向量数据库"""
    print("🔄 重置知识库向量数据库")
    print("=" * 50)
    
    # 1. 删除现有向量数据库
    vector_db_path = "vector_db"
    if os.path.exists(vector_db_path):
        print("🗑️  删除现有向量数据库...")
        shutil.rmtree(vector_db_path)
        print("   ✅ 已删除")
    
    # 2. 重新创建知识库（会自动从JSON加载）
    print("📚 重新创建知识库...")
    kb = SmartKnowledgeBase(data_source="json")
    
    if kb.available:
        doc_count = kb.collection.count()
        print(f"   ✅ 成功创建，包含 {doc_count} 条知识")
        
        # 3. 测试新知识
        print("\n🔍 测试新添加的小麦知识:")
        results = kb.query("小麦播种", "小麦", "播种期")
        
        if results:
            for i, result in enumerate(results, 1):
                if "小麦" in result['content']:
                    print(f"   ✅ 找到小麦知识: {result['content'][:60]}...")
                    break
            else:
                print("   ⚠️  未找到小麦相关知识，可能需要更好的匹配")
        else:
            print("   ❌ 查询无结果")
    else:
        print("   ❌ 知识库创建失败")
    
    print("\n✅ 重置完成！")

if __name__ == "__main__":
    reset_vector_database()