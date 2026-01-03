# 智能知识库检索模块
# 基于ChromaDB和sentence-transformers实现语义搜索
# 支持多种数据源：JSON文件、数据库、硬编码兜底

import os
import sys
from typing import List, Dict, Any
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    SMART_KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    print(f"智能知识库依赖未安装: {e}")
    print("请运行: pip install chromadb sentence-transformers")
    SMART_KNOWLEDGE_AVAILABLE = False

# 导入知识加载器
try:
    from knowledge_loader import KnowledgeLoader
    KNOWLEDGE_LOADER_AVAILABLE = True
except ImportError:
    KNOWLEDGE_LOADER_AVAILABLE = False

class SmartKnowledgeBase:
    """智能知识库检索系统 - 支持多数据源"""
    
    def __init__(self, data_source: str = "json", data_path: str = None):
        """
        初始化智能知识库
        
        Args:
            data_source: 数据源类型 ("json", "database", "hardcoded")
            data_path: 数据文件路径（用于JSON/CSV）
        """
        self.data_source = data_source
        self.data_path = data_path or "data/agriculture_knowledge.json"
        
        if not SMART_KNOWLEDGE_AVAILABLE:
            self.available = False
            return
            
        try:
            # 尝试初始化嵌入模型（轻量级，无需GPU）
            print("正在初始化文本嵌入模型...")
            print("📥 首次使用需要从HuggingFace下载模型文件（约90MB）...")
            
            # 设置更长的超时时间用于模型下载
            import socket
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(60)  # 60秒超时，给下载更多时间
            
            try:
                # 尝试使用镜像源加速下载（如果在中国）
                import os
                os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
                
                print("🔄 正在下载sentence-transformers模型...")
                print("   模型: all-MiniLM-L6-v2 (轻量级文本嵌入模型)")
                print("   大小: ~90MB")
                print("   用途: 将文本转换为向量，实现智能语义搜索")
                
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ 嵌入模型下载并加载成功")
                print("💾 模型已缓存到本地，下次启动将直接使用")
            except Exception as model_error:
                print(f"⚠️ 嵌入模型加载失败: {model_error}")
                print("🔄 切换到离线模式...")
                # 恢复超时设置
                socket.setdefaulttimeout(original_timeout)
                # 使用离线兜底方案
                self._init_offline_mode()
                return
            finally:
                # 恢复原始超时设置
                socket.setdefaulttimeout(original_timeout)
            
            # 创建或连接本地向量数据库
            db_path = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
            os.makedirs(db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="agriculture_knowledge",
                metadata={"description": "农业知识库"}
            )
            
            # 初始化知识库（如果为空或需要更新）
            if self.collection.count() == 0 or self._should_reload_knowledge():
                self._initialize_knowledge_base()
                
            self.available = True
            print("✅ 智能知识库初始化成功")
            
        except Exception as e:
            print(f"⚠️ 智能知识库初始化失败: {e}")
            print("🔄 切换到离线模式...")
            self._init_offline_mode()
    
    def _init_offline_mode(self):
        """初始化离线模式（基于关键词匹配）"""
        self.available = True
        self.offline_mode = True
        self.model = None
        self.client = None
        self.collection = None
        
        # 加载离线知识库
        self.offline_knowledge = self._get_comprehensive_offline_knowledge()
        print("✅ 离线模式初始化成功（基于关键词匹配）")
    
    def _should_reload_knowledge(self) -> bool:
        """检查是否需要重新加载知识库"""
        # 这里可以添加更复杂的逻辑，比如检查文件修改时间
        return False
    
    def _load_knowledge_documents(self) -> List[Dict[str, Any]]:
        """根据配置的数据源加载知识文档"""
        if not KNOWLEDGE_LOADER_AVAILABLE:
            print("知识加载器不可用，使用硬编码知识")
            return self._get_hardcoded_knowledge()
        
        loader = KnowledgeLoader()
        
        if self.data_source == "json":
            return loader.load_from_json(os.path.basename(self.data_path))
        elif self.data_source == "csv":
            return loader.load_from_csv(os.path.basename(self.data_path))
        elif self.data_source == "database":
            # 这里需要传入数据库连接
            # return loader.load_from_database(connection)
            print("数据库加载暂未实现，使用JSON兜底")
            return loader.load_from_json()
        else:
            return self._get_hardcoded_knowledge()

    def _initialize_knowledge_base(self):
        """初始化农业知识库 - 支持多数据源"""
        print(f"正在从 {self.data_source} 初始化农业知识库...")
        
        # 加载知识文档
        knowledge_docs = self._load_knowledge_documents()
        
        if not knowledge_docs:
            print("未找到知识文档，使用硬编码兜底")
            knowledge_docs = self._get_hardcoded_knowledge()
        
        # 清空现有集合（如果需要重新加载）
        try:
            self.client.delete_collection("agriculture_knowledge")
            self.collection = self.client.create_collection(
                name="agriculture_knowledge",
                metadata={"description": "农业知识库"}
            )
        except:
            pass  # 集合可能不存在
        
        # 批量添加文档到向量数据库
        for i, doc in enumerate(knowledge_docs):
            try:
                # 生成文档向量
                embedding = self.model.encode(doc["content"]).tolist()
                
                # 添加到数据库
                self.collection.add(
                    embeddings=[embedding],
                    documents=[doc["content"]],
                    metadatas=[{
                        "source": doc["source"],
                        "crop": doc["crop"], 
                        "stage": doc["stage"],
                        "doc_id": doc.get("id", str(i)),
                        "priority": doc.get("priority", 1)
                    }],
                    ids=[doc.get("id", f"doc_{i}")]
                )
            except Exception as e:
                print(f"添加文档 {i} 失败: {e}")
        
        print(f"成功初始化 {len(knowledge_docs)} 条农业知识")
    
    def _get_hardcoded_knowledge(self) -> List[Dict[str, Any]]:
        """硬编码的兜底知识库（最小集合）"""
        return [
            {
                "id": "hardcoded_leaf_yellow",
                "content": "作物叶片发黄可能的原因：1.缺氮肥导致的生理性黄化；2.根系受损影响养分吸收；3.病害感染如纹枯病、叶枯病；4.虫害危害如蚜虫、红蜘蛛。需要根据具体症状判断原因。",
                "source": "系统内置知识",
                "crop": "通用",
                "stage": "通用",
                "priority": 1
            },
            {
                "id": "hardcoded_pest_control",
                "content": "病虫害综合防治原则：预防为主，综合防治。优先使用农业防治、生物防治，化学防治作为补充。选择高效低毒农药，注意轮换用药避免抗性。",
                "source": "系统内置知识",
                "crop": "通用",
                "stage": "通用",
                "priority": 1
            },
            {
                "id": "hardcoded_drought",
                "content": "高温干旱条件下的应对措施：1.及时灌溉，保持土壤湿润；2.叶面喷水降温；3.覆盖遮阳网或秸秆；4.叶面喷施抗旱剂；5.适当修剪减少蒸腾。",
                "source": "系统内置知识",
                "crop": "通用",
                "stage": "通用",
                "priority": 1
            }
        ]
    
    def _get_comprehensive_offline_knowledge(self) -> List[Dict[str, Any]]:
        """扩展的离线知识库"""
        return [
            # 叶片问题
            {
                "keywords": ["叶子", "叶片", "发黄", "黄化", "变黄"],
                "content": "叶片发黄的常见原因及处理方法：\n1. 缺氮：叶片从下往上发黄，施用氮肥\n2. 缺铁：新叶发黄，叶脉仍绿，喷施铁肥\n3. 病害：叶斑病、纹枯病等，使用杀菌剂\n4. 虫害：蚜虫、红蜘蛛等，使用杀虫剂\n5. 水分：过湿或过干，调节灌溉",
                "crop": "通用",
                "stage": "通用"
            },
            # 病虫害防治
            {
                "keywords": ["病虫害", "防治", "病害", "虫害", "治疗"],
                "content": "病虫害综合防治策略：\n1. 预防为主：选用抗病品种，合理轮作\n2. 农业防治：清洁田园，合理施肥\n3. 生物防治：利用天敌，生物农药\n4. 化学防治：科学用药，轮换使用\n5. 监测预警：定期检查，及时发现",
                "crop": "通用",
                "stage": "通用"
            },
            # 高温干旱
            {
                "keywords": ["高温", "干旱", "缺水", "热害", "抗旱"],
                "content": "高温干旱应对措施：\n1. 灌溉管理：及时补水，滴灌节水\n2. 遮阳降温：搭建遮阳网，减少蒸腾\n3. 叶面喷水：早晚喷水，降低叶温\n4. 覆盖保墒：秸秆覆盖，减少蒸发\n5. 抗旱剂：喷施抗旱剂，提高抗性",
                "crop": "通用",
                "stage": "通用"
            },
            # 施肥管理
            {
                "keywords": ["施肥", "肥料", "营养", "氮磷钾", "追肥"],
                "content": "科学施肥指导：\n1. 基肥：有机肥为主，改良土壤\n2. 追肥：根据生长期需求分次施用\n3. 氮肥：促进茎叶生长，注意用量\n4. 磷肥：促进根系和花果发育\n5. 钾肥：提高抗性，改善品质",
                "crop": "通用",
                "stage": "通用"
            },
            # 水稻专用
            {
                "keywords": ["水稻", "稻田", "分蘖", "抽穗", "灌浆"],
                "content": "水稻管理要点：\n1. 分蘖期：浅水勤灌，促进分蘖\n2. 拔节期：深水护苗，防止倒伏\n3. 抽穗期：保持水层，确保抽穗\n4. 灌浆期：干湿交替，提高品质\n5. 成熟期：适时断水，便于收获",
                "crop": "水稻",
                "stage": "通用"
            },
            # 玉米专用
            {
                "keywords": ["玉米", "拔节", "抽雄", "灌浆", "玉米田"],
                "content": "玉米管理要点：\n1. 苗期：控水蹲苗，促进根系\n2. 拔节期：重施拔节肥，促进茎秆\n3. 抽雄期：保证水分，确保授粉\n4. 灌浆期：充足水肥，提高产量\n5. 成熟期：适时收获，确保品质",
                "crop": "玉米",
                "stage": "通用"
            }
        ]
    
    def add_document(self, content: str, source: str, crop: str = "通用", stage: str = "通用"):
        """添加新的知识文档"""
        if not self.available:
            return False
            
        try:
            # 生成唯一ID
            doc_count = self.collection.count()
            doc_id = f"doc_{doc_count}"
            
            # 生成向量
            embedding = self.model.encode(content).tolist()
            
            # 添加到数据库
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    "source": source,
                    "crop": crop,
                    "stage": stage,
                    "doc_id": doc_count
                }],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            print(f"添加文档失败: {e}")
            return False
    
    def query(self, question: str, crop_type: str = "", growth_stage: str = "", n_results: int = 3) -> List[Dict[str, Any]]:
        """智能查询相关知识"""
        if not self.available:
            return []
        
        # 检查是否为离线模式
        if hasattr(self, 'offline_mode') and self.offline_mode:
            return self._offline_query(question, crop_type, growth_stage, n_results)
            
        try:
            # 构造更具体的查询文本
            query_parts = [question]
            if crop_type:
                query_parts.append(crop_type)
            if growth_stage:
                query_parts.append(growth_stage)
            
            query_text = " ".join(query_parts)
            
            # 生成查询向量
            query_embedding = self.model.encode(query_text).tolist()
            
            # 在向量数据库中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            advice_snippets = []
            if results['documents'] and len(results['documents']) > 0:
                for i, (doc, meta, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    advice_snippets.append({
                        "content": doc,
                        "source": meta.get('source', '未知来源'),
                        "crop": meta.get('crop', '通用'),
                        "stage": meta.get('stage', '通用'),
                        "relevance_score": max(0, 1 - distance),  # 确保分数为正数
                        "distance": distance,  # 保留原始距离用于调试
                        "rank": i + 1
                    })
            
            return advice_snippets
            
        except Exception as e:
            print(f"智能查询失败: {e}")
            # 如果向量查询失败，回退到离线模式
            return self._offline_query(question, crop_type, growth_stage, n_results)
    
    def _offline_query(self, question: str, crop_type: str = "", growth_stage: str = "", n_results: int = 3) -> List[Dict[str, Any]]:
        """离线关键词匹配查询"""
        question_lower = question.lower()
        crop_lower = crop_type.lower()
        
        matched_results = []
        
        for knowledge in self.offline_knowledge:
            score = 0
            
            # 关键词匹配
            for keyword in knowledge["keywords"]:
                if keyword in question_lower:
                    score += 2
            
            # 作物匹配
            if crop_lower and knowledge["crop"].lower() in [crop_lower, "通用"]:
                score += 1
            elif knowledge["crop"] == "通用":
                score += 0.5
            
            if score > 0:
                matched_results.append({
                    "content": knowledge["content"],
                    "source": "离线知识库",
                    "crop": knowledge["crop"],
                    "stage": knowledge["stage"],
                    "relevance_score": min(score / 3, 1.0),  # 归一化到0-1
                    "distance": 1 - min(score / 3, 1.0),
                    "rank": 0
                })
        
        # 按相关度排序
        matched_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 重新设置排名
        for i, result in enumerate(matched_results[:n_results]):
            result["rank"] = i + 1
        
        return matched_results[:n_results]

    
    def format_advice(self, snippets: List[Dict[str, Any]], question: str) -> str:
        """将检索结果格式化为友好的建议文本"""
        if not snippets:
            return "抱歉，未找到相关的农业知识。建议咨询当地农技专家。"
        
        formatted = f"根据农业知识库，关于「{question}」的建议如下：\n\n"
        
        for snippet in snippets:
            relevance = snippet.get('relevance_score', 0)
            distance = snippet.get('distance', 0)
            # 对于向量搜索，距离小于5.0通常表示有一定相关性
            if distance < 5.0:  
                formatted += f"💡 {snippet['content']}\n"
                formatted += f"   📚 来源：{snippet['source']}\n\n"
        
        formatted += "---\n"
        formatted += "*以上建议基于权威农业资料，请结合实地情况灵活应用。如有疑问，建议咨询当地农技专家。*"
        
        return formatted

# 全局实例
smart_kb = None

def get_smart_knowledge_base(data_source: str = "json", data_path: str = None):
    """获取智能知识库实例（单例模式）"""
    global smart_kb
    if smart_kb is None:
        smart_kb = SmartKnowledgeBase(data_source=data_source, data_path=data_path)
    return smart_kb

def smart_query(question: str, crop_type: str = "", growth_stage: str = "") -> str:
    """便捷的智能查询函数"""
    kb = get_smart_knowledge_base()  # 默认使用JSON数据源
    if not kb.available:
        return "智能知识库暂不可用，请检查相关依赖是否已安装。"
    
    snippets = kb.query(question, crop_type, growth_stage)
    return kb.format_advice(snippets, question)

if __name__ == "__main__":
    # 测试智能知识库
    print("测试智能知识库...")
    
    kb = SmartKnowledgeBase()
    if kb.available:
        # 测试查询
        test_queries = [
            ("叶子发黄怎么办", "水稻", "分蘖期"),
            ("如何施肥", "玉米", "拔节期"),
            ("病虫害防治", "", ""),
            ("高温干旱", "", "")
        ]
        
        for question, crop, stage in test_queries:
            print(f"\n查询: {question} (作物: {crop}, 阶段: {stage})")
            result = smart_query(question, crop, stage)
            print(result)
            print("-" * 50)
    else:
        print("智能知识库不可用")