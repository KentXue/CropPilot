# 文件：app.py
from flask import Flask, request, jsonify, render_template
from decision_engine import get_suggestions
from database import get_connection
from datetime import datetime
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.settings import Config

# 尝试导入CORS，如果未安装则跳过
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config.from_object(Config)

if CORS_AVAILABLE:
    CORS(app)  # 允许跨域请求


@app.route('/api/get_advice', methods=['GET'])
def api_get_advice():
    """提供决策建议的API接口"""
    # 从URL参数中获取用户选择，例如：/api/get_advice?crop=水稻&stage=分蘖期
    crop = request.args.get('crop', '水稻')
    stage = request.args.get('stage', '分蘖期')

    # 调用决策引擎
    advice_list = get_suggestions(crop, stage)

    # 保存决策记录到数据库
    try:
        save_decision_record(crop, stage, '\n'.join(advice_list))
    except Exception as e:
        print(f"保存决策记录失败: {e}")

    # 以JSON格式返回结果
    return jsonify({
        "crop": crop,
        "stage": stage,
        "advice": advice_list,
        "status": "success"
    })


@app.route('/api/save_sensor_data', methods=['POST'])
def api_save_sensor_data():
    """保存传感器数据"""
    try:
        data = request.json
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO sensor_data 
                    (crop_type, growth_stage, temperature, humidity, soil_moisture, 
                     light_intensity, ph_value, nitrogen, phosphorus, potassium, location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    data.get('crop_type'),
                    data.get('growth_stage'),
                    data.get('temperature'),
                    data.get('humidity'),
                    data.get('soil_moisture'),
                    data.get('light_intensity'),
                    data.get('ph_value'),
                    data.get('nitrogen'),
                    data.get('phosphorus'),
                    data.get('potassium'),
                    data.get('location')
                ))
                conn.commit()
                data_id = cursor.lastrowid
                return jsonify({
                    "status": "success",
                    "data_id": data_id,
                    "message": "传感器数据保存成功"
                })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/get_sensor_data', methods=['GET'])
def api_get_sensor_data():
    """查询传感器数据"""
    try:
        crop = request.args.get('crop', None)
        stage = request.args.get('stage', None)
        limit = int(request.args.get('limit', 50))
        
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM sensor_data WHERE 1=1"
                params = []
                
                if crop:
                    sql += " AND crop_type = %s"
                    params.append(crop)
                if stage:
                    sql += " AND growth_stage = %s"
                    params.append(stage)
                
                sql += " ORDER BY recorded_at DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(sql, params)
                records = cursor.fetchall()
                
                # 转换datetime对象为字符串
                for record in records:
                    if 'recorded_at' in record and record['recorded_at']:
                        record['recorded_at'] = record['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({
                    "status": "success",
                    "count": len(records),
                    "data": records
                })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/get_history', methods=['GET'])
def api_get_history():
    """查询决策历史记录"""
    try:
        crop = request.args.get('crop', None)
        stage = request.args.get('stage', None)
        limit = int(request.args.get('limit', 50))
        
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM decision_records WHERE 1=1"
                params = []
                
                if crop:
                    sql += " AND crop_type = %s"
                    params.append(crop)
                if stage:
                    sql += " AND growth_stage = %s"
                    params.append(stage)
                
                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(sql, params)
                records = cursor.fetchall()
                
                # 转换datetime对象为字符串
                for record in records:
                    if 'created_at' in record and record['created_at']:
                        record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                return jsonify({
                    "status": "success",
                    "count": len(records),
                    "records": records
                })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def save_decision_record(crop_type, growth_stage, advice, sensor_data_id=None):
    """保存决策记录到数据库"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO decision_records 
                    (crop_type, growth_stage, sensor_data_id, advice)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (crop_type, growth_stage, sensor_data_id, advice))
                conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"保存决策记录失败: {e}")


@app.route('/')
def serve_index():
    """
    当用户访问根路径（如 http://127.0.0.1:5000/）时，
    返回 index.html 模板。
    """
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], port=5000, host='0.0.0.0')

