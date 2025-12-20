#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识库加载器
支持多种数据源：JSON文件、数据库、CSV等
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime

class KnowledgeLoader:
    """知识库数据加载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.knowledge_cache = {}
        
    def load_from_json(self, filename: str = "agriculture_knowledge.json") -> List[Dict[str, Any]]:
        """从JSON文件加载知识库"""
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"知识库文件不存在: {file_path}")
            return self._get_fallback_knowledge()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = data.get('knowledge_base', {}).get('documents', [])
            
            # 过滤活跃的文档并按优先级排序
            active_docs = [doc for doc in documents if doc.get('active', True)]
            active_docs.sort(key=lambda x: x.get('priority', 999))
            
            print(f"从 {filename} 加载了 {len(active_docs)} 条知识")
            return active_docs
            
        except Exception as e:
            print(f"加载知识库失败: {e}")
            return self._get_fallback_knowledge()
    
    def load_from_csv(self, filename: str = "agriculture_knowledge.csv") -> List[Dict[str, Any]]:
        """从CSV文件加载知识库"""
        file_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"CSV文件不存在: {file_path}")
            return []
        
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('active', 'true').lower() == 'true':
                        # 处理关键词字段
                        keywords = row.get('keywords', '').split(',') if row.get('keywords') else []
                        keywords = [k.strip() for k in keywords if k.strip()]
                        
                        doc = {
                            'id': row.get('id', ''),
                            'content': row.get('content', ''),
                            'source': row.get('source', ''),
                            'crop': row.get('crop', '通用'),
                            'stage': row.get('stage', '通用'),
                            'keywords': keywords,
                            'priority': int(row.get('priority', 1)),
                            'active': True
                        }
                        documents.append(doc)
            
            print(f"从 {filename} 加载了 {len(documents)} 条知识")
            return documents
            
        except Exception as e:
            print(f"加载CSV失败: {e}")
            return []
    
    def load_from_database(self, connection) -> List[Dict[str, Any]]:
        """从数据库加载知识库"""
        try:
            with connection.cursor() as cursor:
                sql = """
                    SELECT id, content, source, crop, stage, keywords, priority
                    FROM knowledge_documents 
                    WHERE active = TRUE 
                    ORDER BY priority ASC, created_at DESC
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                documents = []
                for row in rows:
                    keywords = row.get('keywords', '').split(',') if row.get('keywords') else []
                    keywords = [k.strip() for k in keywords if k.strip()]
                    
                    doc = {
                        'id': row['id'],
                        'content': row['content'],
                        'source': row['source'],
                        'crop': row['crop'],
                        'stage': row['stage'],
                        'keywords': keywords,
                        'priority': row['priority'],
                        'active': True
                    }
                    documents.append(doc)
                
                print(f"从数据库加载了 {len(documents)} 条知识")
                return documents
                
        except Exception as e:
            print(f"从数据库加载知识失败: {e}")
            return []
    
    def save_to_json(self, documents: List[Dict[str, Any]], filename: str = "agriculture_knowledge.json"):
        """保存知识库到JSON文件"""
        file_path = os.path.join(self.data_dir, filename)
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        data = {
            "knowledge_base": {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "documents": documents
            }
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"知识库已保存到 {filename}")
            return True
        except Exception as e:
            print(f"保存知识库失败: {e}")
            return False
    
    def add_document(self, content: str, source: str, crop: str = "通用", 
                    stage: str = "通用", keywords: List[str] = None, 
                    priority: int = 1) -> Dict[str, Any]:
        """添加新的知识文档"""
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        document = {
            'id': doc_id,
            'content': content,
            'source': source,
            'crop': crop,
            'stage': stage,
            'keywords': keywords or [],
            'priority': priority,
            'active': True
        }
        
        return document
    
    def _get_fallback_knowledge(self) -> List[Dict[str, Any]]:
        """获取兜底的硬编码知识（最小集合）"""
        return [
            {
                'id': 'fallback_1',
                'content': '作物叶片发黄可能的原因：1.缺氮肥导致的生理性黄化；2.根系受损影响养分吸收；3.病害感染；4.虫害危害。需要根据具体症状判断原因。',
                'source': '系统内置知识',
                'crop': '通用',
                'stage': '通用',
                'keywords': ['叶片发黄', '缺氮', '病害', '虫害'],
                'priority': 1,
                'active': True
            },
            {
                'id': 'fallback_2',
                'content': '病虫害综合防治原则：预防为主，综合防治。优先使用农业防治、生物防治，化学防治作为补充。',
                'source': '系统内置知识',
                'crop': '通用',
                'stage': '通用',
                'keywords': ['病虫害', '综合防治', '预防'],
                'priority': 1,
                'active': True
            }
        ]

# 使用示例
if __name__ == "__main__":
    loader = KnowledgeLoader()
    
    # 测试加载JSON
    docs = loader.load_from_json()
    print(f"加载了 {len(docs)} 条知识")
    
    # 测试添加新文档
    new_doc = loader.add_document(
        content="测试知识：这是一条新的农业知识。",
        source="测试来源",
        crop="测试作物",
        keywords=["测试", "新知识"]
    )
    print(f"新文档: {new_doc['id']}")