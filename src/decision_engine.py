from database import get_connection
from knowledge_base import knowledge_base

# 尝试导入智能知识库
try:
    from smart_knowledge import get_smart_knowledge_base, smart_query
    SMART_KNOWLEDGE_AVAILABLE = True
except ImportError:
    print("智能知识库模块未找到，将使用传统规则引擎")
    SMART_KNOWLEDGE_AVAILABLE = False

def get_suggestions(crop_type, growth_stage, sensor_values=None):
    """
    获取作物决策建议（三级决策：数据库规则 -> 智能检索 -> 硬编码规则）
    """
    # 第一级：尝试从数据库获取精确匹配的规则
    try:
        suggestions = get_suggestions_from_db(crop_type, growth_stage, sensor_values)
        if suggestions:
            # 将数据库返回的字典格式转换为字符串列表
            result = []
            for item in suggestions:
                result.append(f"{item['type']}: {item['action']}")
            return result
    except Exception as e:
        print(f"数据库查询失败: {e}")
    
    # 第二级：如果数据库没有精确匹配，尝试智能检索
    if SMART_KNOWLEDGE_AVAILABLE:
        try:
            kb = get_smart_knowledge_base()
            if kb.available:
                # 构造查询问题
                query_question = f"{crop_type}{growth_stage}管理建议"
                
                # 如果有传感器数据，添加到查询中
                if sensor_values:
                    conditions = []
                    if sensor_values.get('temperature'):
                        temp = float(sensor_values['temperature'])
                        if temp > 30:
                            conditions.append("高温")
                        elif temp < 15:
                            conditions.append("低温")
                    
                    if sensor_values.get('humidity'):
                        humidity = float(sensor_values['humidity'])
                        if humidity > 80:
                            conditions.append("高湿")
                        elif humidity < 40:
                            conditions.append("干燥")
                    
                    if conditions:
                        query_question += " " + " ".join(conditions) + "条件"
                
                # 执行智能查询
                snippets = kb.query(query_question, crop_type, growth_stage, n_results=2)
                if snippets:
                    result = []
                    for snippet in snippets:
                        if snippet.get('relevance_score', 0) > 0.3:  # 只使用相关度较高的结果
                            result.append(f"智能建议: {snippet['content'][:100]}...")
                    
                    if result:
                        result.append("*以上建议来自智能知识库，请结合实际情况应用")
                        return result
        except Exception as e:
            print(f"智能检索失败: {e}")
    
    # 第三级：兜底使用硬编码知识库
    result = []
    if crop_type in knowledge_base:
        crop_rules = knowledge_base[crop_type]
        for rule_type, rules in crop_rules.items():
            for rule in rules:
                if rule.get("生长阶段") == growth_stage:
                    result.append(f"{rule_type}: {rule.get('建议', '')}")
    
    return result if result else [f"暂无{crop_type}在{growth_stage}阶段的建议"]

def get_smart_advice(question, crop_type="", growth_stage=""):
    """
    直接使用智能检索获取建议（新增接口）
    """
    if not SMART_KNOWLEDGE_AVAILABLE:
        return "智能建议功能暂不可用，请安装相关依赖包。"
    
    try:
        return smart_query(question, crop_type, growth_stage)
    except Exception as e:
        print(f"智能建议获取失败: {e}")
        return "智能建议获取失败，请稍后重试。"

def get_suggestions_from_db(crop_type, growth_stage, sensor_values=None):
    """从数据库知识库中查询建议"""
    suggestions = []
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 查询该作物、该阶段的所有有效规则
            sql = """
                SELECT rule_type, action, conditions 
                FROM knowledge_rules 
                WHERE crop_type = %s 
                  AND growth_stage = %s 
                  AND is_active = TRUE
                ORDER BY priority DESC
            """
            cursor.execute(sql, (crop_type, growth_stage))
            rules = cursor.fetchall()
            
            for rule in rules:
                # 如果有条件，可以在这里解析 rule['conditions'] (JSON格式)
                # 并与传入的 sensor_values 进行匹配判断
                # if check_conditions(rule['conditions'], sensor_values):
                suggestions.append({
                    'type': rule['rule_type'],
                    'action': rule['action']
                })
    finally:
        conn.close()
    
    return suggestions

