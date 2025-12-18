from database import get_connection
from knowledge_base import knowledge_base

def get_suggestions(crop_type, growth_stage, sensor_values=None):
    """
    获取作物决策建议（优先从数据库，如果数据库没有则从知识库）
    """
    # 首先尝试从数据库获取
    try:
        suggestions = get_suggestions_from_db(crop_type, growth_stage, sensor_values)
        if suggestions:
            # 将数据库返回的字典格式转换为字符串列表
            result = []
            for item in suggestions:
                result.append(f"{item['type']}: {item['action']}")
            return result
    except Exception as e:
        print(f"数据库查询失败，使用知识库: {e}")
    
    # 如果数据库没有数据，从知识库获取
    result = []
    if crop_type in knowledge_base:
        crop_rules = knowledge_base[crop_type]
        for rule_type, rules in crop_rules.items():
            for rule in rules:
                if rule.get("生长阶段") == growth_stage:
                    result.append(f"{rule_type}: {rule.get('建议', '')}")
    
    return result if result else [f"暂无{crop_type}在{growth_stage}阶段的建议"]

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

