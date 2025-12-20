#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
农业知识库管理工具
用于管理JSON格式的知识库文件
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from knowledge_loader import KnowledgeLoader

class KnowledgeManager:
    """知识库管理器"""
    
    def __init__(self, json_file: str = "data/agriculture_knowledge.json"):
        self.json_file = json_file
        self.loader = KnowledgeLoader()
        
    def list_knowledge(self):
        """列出所有知识条目"""
        print("📚 当前知识库内容:")
        print("=" * 80)
        
        docs = self.loader.load_from_json(os.path.basename(self.json_file))
        
        if not docs:
            print("❌ 知识库为空或加载失败")
            return
        
        for i, doc in enumerate(docs, 1):
            status = "✅" if doc.get('active', True) else "❌"
            priority = doc.get('priority', 1)
            
            print(f"{i:2d}. {status} [{doc['crop']}-{doc['stage']}] (优先级:{priority})")
            print(f"    ID: {doc.get('id', 'N/A')}")
            print(f"    内容: {doc['content'][:60]}...")
            print(f"    来源: {doc['source']}")
            print()
    
    def add_knowledge(self, content: str, source: str, crop: str = "通用", 
                     stage: str = "通用", keywords: List[str] = None, 
                     priority: int = 1):
        """添加新的知识条目"""
        
        # 加载现有数据
        file_path = self.json_file
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {
                "knowledge_base": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "documents": []
                }
            }
        
        # 生成新ID
        existing_ids = [doc.get('id', '') for doc in data['knowledge_base']['documents']]
        doc_id = f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建新文档
        new_doc = {
            "id": doc_id,
            "content": content,
            "source": source,
            "crop": crop,
            "stage": stage,
            "keywords": keywords or [],
            "priority": priority,
            "active": True
        }
        
        # 添加到数据
        data['knowledge_base']['documents'].append(new_doc)
        data['knowledge_base']['last_updated'] = datetime.now().isoformat()
        
        # 保存文件
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 成功添加知识条目: {doc_id}")
        return doc_id
    
    def disable_knowledge(self, doc_id: str):
        """禁用知识条目"""
        return self._update_knowledge_status(doc_id, False)
    
    def enable_knowledge(self, doc_id: str):
        """启用知识条目"""
        return self._update_knowledge_status(doc_id, True)
    
    def _update_knowledge_status(self, doc_id: str, active: bool):
        """更新知识条目状态"""
        file_path = self.json_file
        
        if not os.path.exists(file_path):
            print("❌ 知识库文件不存在")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 查找并更新文档
        found = False
        for doc in data['knowledge_base']['documents']:
            if doc.get('id') == doc_id:
                doc['active'] = active
                found = True
                break
        
        if not found:
            print(f"❌ 未找到ID为 {doc_id} 的知识条目")
            return False
        
        # 保存文件
        data['knowledge_base']['last_updated'] = datetime.now().isoformat()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        status_text = "启用" if active else "禁用"
        print(f"✅ 成功{status_text}知识条目: {doc_id}")
        return True
    
    def search_knowledge(self, keyword: str):
        """搜索知识条目"""
        print(f"🔍 搜索关键词: {keyword}")
        print("=" * 60)
        
        docs = self.loader.load_from_json(os.path.basename(self.json_file))
        
        matches = []
        for doc in docs:
            if (keyword.lower() in doc['content'].lower() or 
                keyword.lower() in doc['source'].lower() or
                keyword.lower() in doc['crop'].lower() or
                keyword.lower() in doc['stage'].lower() or
                any(keyword.lower() in kw.lower() for kw in doc.get('keywords', []))):
                matches.append(doc)
        
        if not matches:
            print("❌ 未找到匹配的知识条目")
            return
        
        for i, doc in enumerate(matches, 1):
            status = "✅" if doc.get('active', True) else "❌"
            print(f"{i}. {status} [{doc['crop']}-{doc['stage']}]")
            print(f"   ID: {doc.get('id', 'N/A')}")
            print(f"   内容: {doc['content'][:80]}...")
            print()

def main():
    """主函数 - 命令行界面"""
    manager = KnowledgeManager()
    
    while True:
        print("\n" + "=" * 60)
        print("🌾 农业知识库管理工具")
        print("=" * 60)
        print("1. 查看所有知识")
        print("2. 添加新知识")
        print("3. 搜索知识")
        print("4. 禁用知识")
        print("5. 启用知识")
        print("0. 退出")
        print("-" * 60)
        
        choice = input("请选择操作 (0-5): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            manager.list_knowledge()
        elif choice == "2":
            print("\n📝 添加新知识:")
            content = input("知识内容: ").strip()
            source = input("知识来源: ").strip()
            crop = input("适用作物 (默认:通用): ").strip() or "通用"
            stage = input("生长阶段 (默认:通用): ").strip() or "通用"
            keywords_str = input("关键词 (用逗号分隔): ").strip()
            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()] if keywords_str else []
            
            try:
                priority = int(input("优先级 (1-5, 默认:1): ").strip() or "1")
            except ValueError:
                priority = 1
            
            if content and source:
                manager.add_knowledge(content, source, crop, stage, keywords, priority)
            else:
                print("❌ 内容和来源不能为空")
        elif choice == "3":
            keyword = input("搜索关键词: ").strip()
            if keyword:
                manager.search_knowledge(keyword)
        elif choice == "4":
            doc_id = input("要禁用的知识ID: ").strip()
            if doc_id:
                manager.disable_knowledge(doc_id)
        elif choice == "5":
            doc_id = input("要启用的知识ID: ").strip()
            if doc_id:
                manager.enable_knowledge(doc_id)
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main()