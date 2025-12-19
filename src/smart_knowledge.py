# 智能知识库检索模块
# 基于ChromaDB和sentence-transformers实现语义搜索

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

class SmartKnowledgeBase:
    """智能知识库检索系统"""
    
    def __init__(self):
        if not SMART_KNOWLEDGE_AVAILABLE:
            self.available = False
            return
            
        try:
            # 初始化嵌入模型（轻量级，无需GPU）
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 创建或连接本地向量数据库
            db_path = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
            os.makedirs(db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="agriculture_knowledge",
                metadata={"description": "农业知识库"}
            )
            
            # 初始化知识库（如果为空）
            if self.collection.count() == 0:
                self._initialize_knowledge_base()
                
            self.available = True
            print("智能知识库初始化成功")
            
        except Exception as e:
            print(f"智能知识库初始化失败: {e}")
            self.available = False
    
    def _initialize_knowledge_base(self):
        """初始化农业知识库"""
        print("正在初始化农业知识库...")
        
        # 农业知识文档
        knowledge_docs = [
            {
                "content": "水稻分蘖期是水稻生长的关键时期，此时需要保持浅水层3-5cm，促进分蘖。施肥方面，每亩追施尿素5-8公斤，促进分蘖发生。注意防治稻飞虱和纹枯病。",
                "source": "水稻栽培技术手册",
                "crop": "水稻",
                "stage": "分蘖期"
            },
            {
                "content": "水稻拔节期要控制氮肥施用，防止徒长倒伏。保持适度水层，避免过深或过浅。此期是决定穗数的关键期，要加强田间管理。",
                "source": "水稻栽培技术手册", 
                "crop": "水稻",
                "stage": "拔节期"
            },
            {
                "content": "水稻抽穗期需要充足的水分供应，保持水层5-7cm。叶面喷施磷酸二氢钾，提高结实率。注意防治稻瘟病和褐飞虱。",
                "source": "水稻栽培技术手册",
                "crop": "水稻", 
                "stage": "抽穗期"
            },
            {
                "content": "玉米苗期管理要点：保持土壤湿润但不积水，基肥为主，可适当追施少量氮肥。注意防治地下害虫如蛴螬、金针虫等。",
                "source": "玉米栽培技术指南",
                "crop": "玉米",
                "stage": "苗期"
            },
            {
                "content": "玉米拔节期是需水需肥的关键期，追施氮肥促进茎秆生长。保持充足水分，但要注意排水防涝。此期要防治玉米螟虫害。",
                "source": "玉米栽培技术指南",
                "crop": "玉米", 
                "stage": "拔节期"
            },
            {
                "content": "玉米抽雄期需要大量水分，是决定产量的关键期。增施磷钾肥，促进授粉结实。注意防治玉米大斑病和小斑病。",
                "source": "玉米栽培技术指南",
                "crop": "玉米",
                "stage": "抽雄期"
            },
            {
                "content": "作物叶片发黄可能的原因：1.缺氮肥导致的生理性黄化；2.根系受损影响养分吸收；3.病害感染如纹枯病、叶枯病；4.虫害危害如蚜虫、红蜘蛛。需要根据具体症状判断原因。",
                "source": "作物病虫害诊断手册",
                "crop": "通用",
                "stage": "通用"
            },
            {
                "content": "土壤pH值过高或过低都会影响作物生长。pH值6.0-7.0最适宜大多数作物。pH过低可施用石灰调节，pH过高可施用硫磺或有机肥改良。",
                "source": "土壤改良技术手册", 
                "crop": "通用",
                "stage": "通用"
            },
            {
                "content": "高温干旱条件下的应对措施：1.及时灌溉，保持土壤湿润；2.叶面喷水降温；3.覆盖遮阳网或秸秆；4.叶面喷施抗旱剂；5.适当修剪减少蒸腾。",
                "source": "农业气象灾害防御手册",
                "crop": "通用", 
                "stage": "通用"
            },
            {
                "content": "病虫害综合防治原则：预防为主，综合防治。优先使用农业防治、生物防治，化学防治作为补充。选择高效低毒农药，注意轮换用药避免抗性。",
                "source": "病虫害防治指南",
                "crop": "通用",
                "stage": "通用"
            }
        ]
        
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
                        "doc_id": i
                    }],
                    ids=[f"doc_{i}"]
                )
            except Exception as e:
                print(f"添加文档 {i} 失败: {e}")
        
        print(f"成功初始化 {len(knowledge_docs)} 条农业知识")
    
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
                        "relevance_score": 1 - distance,  # 转换为相似度分数
                        "rank": i + 1
                    })
            
            return advice_snippets
            
        except Exception as e:
            print(f"智能查询失败: {e}")
            return []
    
    def format_advice(self, snippets: List[Dict[str, Any]], question: str) -> str:
        """将检索结果格式化为友好的建议文本"""
        if not snippets:
            return "抱歉，未找到相关的农业知识。建议咨询当地农技专家。"
        
        formatted = f"根据农业知识库，关于「{question}」的建议如下：\n\n"
        
        for snippet in snippets:
            relevance = snippet.get('relevance_score', 0)
            if relevance > 0.3:  # 只显示相关度较高的结果
                formatted += f"💡 {snippet['content']}\n"
                formatted += f"   📚 来源：{snippet['source']}\n\n"
        
        formatted += "---\n"
        formatted += "*以上建议基于权威农业资料，请结合实地情况灵活应用。如有疑问，建议咨询当地农技专家。*"
        
        return formatted

# 全局实例
smart_kb = None

def get_smart_knowledge_base():
    """获取智能知识库实例（单例模式）"""
    global smart_kb
    if smart_kb is None:
        smart_kb = SmartKnowledgeBase()
    return smart_kb

def smart_query(question: str, crop_type: str = "", growth_stage: str = "") -> str:
    """便捷的智能查询函数"""
    kb = get_smart_knowledge_base()
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