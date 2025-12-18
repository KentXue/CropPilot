# 文件：database.py
import pymysql
import json
from datetime import datetime
import os
import sys

# 添加项目根目录到路径，以便导入config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import Config

# 数据库配置（从配置文件读取）
DB_CONFIG = {
    'host': Config.DB_HOST,
    'user': Config.DB_USER,
    'password': Config.DB_PASSWORD,
    'database': Config.DB_NAME,
    'charset': Config.DB_CHARSET,
    'cursorclass': pymysql.cursors.DictCursor  # 返回字典格式结果
}

def get_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)

# 初始化建表函数（只需运行一次）
def init_tables():
    """初始化数据库表结构"""
    # 获取sql目录的路径
    sql_dir = os.path.join(os.path.dirname(__file__), '..', 'sql')
    schema_file = os.path.join(sql_dir, 'schema.sql')
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 读取并执行schema.sql
            with open(schema_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
                # 分割SQL语句（以分号分隔，但要注意字符串中的分号）
                statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]
                for statement in statements:
                    if statement:
                        try:
                            cursor.execute(statement)
                        except Exception as e:
                            # 忽略某些错误（如表已存在等）
                            if 'already exists' not in str(e).lower():
                                print(f"执行SQL时出错: {e}")
            conn.commit()
            print("所有表创建成功！")
    except Exception as e:
        print(f"初始化表结构失败: {e}")
        conn.rollback()
    finally:
        conn.close()

