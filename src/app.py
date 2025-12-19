# 文件：app.py
from flask import Flask, request, jsonify, render_template
from decision_engine import get_suggestions, get_smart_advice
from database import get_connection
from datetime import datetime
from werkzeug.utils import secure_filename
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

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

if CORS_AVAILABLE:
    CORS(app)  # 允许跨域请求


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


@app.route('/api/get_advice', methods=['GET'])
def api_get_advice():
    """提供决策建议的API接口（支持按地块获取建议）"""
    # 支持两种调用方式：
    # 1) 传 field_id（推荐）：/api/get_advice?field_id=1
    # 2) 兼容旧方式：/api/get_advice?crop=水稻&stage=分蘖期
    field_id = request.args.get('field_id', type=int)
    crop = request.args.get('crop')
    stage = request.args.get('stage', '分蘖期')

    # 如果传入了 field_id，则优先根据地块信息确定作物类型
    if field_id is not None:
        try:
            conn = get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT crop_type FROM fields WHERE id = %s",
                        (field_id,)
                    )
                    field = cursor.fetchone()
                    if field:
                        crop = crop or field.get('crop_type')
            finally:
                conn.close()
        except Exception as e:
            print(f"根据 field_id 获取作物信息失败: {e}")

    # 兜底：如果依然没有 crop，就退回到默认值
    if not crop:
        crop = '水稻'

    # 调用决策引擎
    advice_list = get_suggestions(crop, stage)

    # 保存决策记录到数据库（如果没有 field_id，则允许为空，便于兼容旧数据）
    try:
        save_decision_record(crop, stage, '\n'.join(advice_list), field_id=field_id)
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
                    (field_id, crop_type, growth_stage, temperature, humidity, soil_moisture, 
                     light_intensity, ph_value, nitrogen, phosphorus, potassium, location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                field_id = data.get('field_id')
                if field_id is None:
                    raise ValueError("缺少必须参数 field_id")

                crop_type = data.get('crop_type')
                # 如果未显式传入 crop_type，则尝试根据 field_id 从 fields 表中获取
                if not crop_type:
                    try:
                        cursor.execute(
                            "SELECT crop_type FROM fields WHERE id = %s",
                            (field_id,)
                        )
                        field = cursor.fetchone()
                        if field:
                            crop_type = field.get('crop_type')
                    except Exception as e:
                        print(f"根据 field_id 获取作物类型失败: {e}")

                growth_stage = data.get('growth_stage')

                cursor.execute(sql, (
                    field_id,
                    crop_type,
                    growth_stage,
                    data.get('crop_type'),
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


@app.route('/api/fields', methods=['GET'])
def api_get_fields():
    """获取地块列表，可按用户过滤"""
    try:
        user_id = request.args.get('user_id', type=int)
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM fields WHERE 1=1"
                params = []
                if user_id is not None:
                    sql += " AND user_id = %s"
                    params.append(user_id)
                sql += " ORDER BY id ASC"
                cursor.execute(sql, params)
                fields = cursor.fetchall()
                return jsonify({
                    "status": "success",
                    "count": len(fields),
                    "fields": fields
                })
        finally:
            conn.close()
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/users', methods=['GET'])
def api_get_users():
    """获取用户列表（演示用，不含敏感信息）"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, role, phone, created_at FROM users ORDER BY id ASC"
                )
                users = cursor.fetchall()
                return jsonify({
                    "status": "success",
                    "count": len(users),
                    "users": users
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
        field_id = request.args.get('field_id', type=int)
        crop = request.args.get('crop', None)
        stage = request.args.get('stage', None)
        limit = int(request.args.get('limit', 50))
        
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM sensor_data WHERE 1=1"
                params = []

                if field_id is not None:
                    sql += " AND field_id = %s"
                    params.append(field_id)
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
        field_id = request.args.get('field_id', type=int)
        crop = request.args.get('crop', None)
        stage = request.args.get('stage', None)
        limit = int(request.args.get('limit', 50))
        
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM decision_records WHERE 1=1"
                params = []

                if field_id is not None:
                    sql += " AND field_id = %s"
                    params.append(field_id)
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


def save_decision_record(crop_type, growth_stage, advice, sensor_data_id=None, field_id=None):
    """保存决策记录到数据库"""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO decision_records 
                    (field_id, crop_type, growth_stage, sensor_data_id, advice)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (field_id, crop_type, growth_stage, sensor_data_id, advice))
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


@app.route('/api/smart_advice', methods=['POST'])
def api_smart_advice():
    """智能农事建议API（基于自然语言查询）"""
    try:
        data = request.json
        question = data.get('question', '')
        crop_type = data.get('crop_type', '')
        growth_stage = data.get('growth_stage', '')
        
        if not question:
            return jsonify({
                "status": "error",
                "message": "请提供查询问题"
            }), 400
        
        # 调用智能建议引擎
        advice = get_smart_advice(question, crop_type, growth_stage)
        
        return jsonify({
            "status": "success",
            "question": question,
            "crop_type": crop_type,
            "growth_stage": growth_stage,
            "advice": advice
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/upload_crop_image', methods=['POST'])
def api_upload_crop_image():
    """上传作物图片并保存路径，可选进行病害识别"""
    try:
        field_id = request.form.get('field_id', type=int)
        if field_id is None:
            return jsonify({"status": "error", "message": "缺少必须参数 field_id"}), 400

        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "请上传图片文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "文件名不能为空"}), 400

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "不支持的文件类型"}), 400

        upload_folder = app.config.get('UPLOAD_FOLDER')
        os.makedirs(upload_folder, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        safe_name = secure_filename(file.filename)
        final_filename = f"{field_id}_{timestamp}_{safe_name}"
        saved_path = os.path.join(upload_folder, final_filename)
        file.save(saved_path)

        captured_at_raw = request.form.get('captured_at')
        captured_at = None
        if captured_at_raw:
            try:
                captured_at = datetime.fromisoformat(captured_at_raw)
            except ValueError:
                captured_at = None

        # 将路径保存为相对路径，便于前端引用
        relative_path = os.path.relpath(
            saved_path,
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        ).replace(os.sep, '/')

        # 获取地块信息以确定作物类型
        crop_type = ""
        try:
            conn = get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT crop_type FROM fields WHERE id = %s", (field_id,))
                    field = cursor.fetchone()
                    if field:
                        crop_type = field.get('crop_type', '')
            finally:
                conn.close()
        except Exception as e:
            print(f"获取地块信息失败: {e}")

        # 尝试进行图像识别
        recognition_result = None
        try:
            from image_recognition import analyze_crop_image
            recognition_result = analyze_crop_image(saved_path, crop_type)
        except ImportError:
            print("图像识别模块未找到")
        except Exception as e:
            print(f"图像识别失败: {e}")

        # 保存到数据库
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO crop_images (field_id, image_path, captured_at)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (field_id, relative_path, captured_at))
                conn.commit()
                image_id = cursor.lastrowid
        finally:
            conn.close()

        response_data = {
            "status": "success",
            "image_id": image_id,
            "image_path": f"/{relative_path}"
        }

        # 如果图像识别成功，添加识别结果
        if recognition_result and recognition_result.get('status') == 'success':
            response_data["recognition"] = recognition_result["analysis_result"]
            
            # 如果识别出病害，自动生成防治建议
            if recognition_result["analysis_result"]["disease_name"] != "健康状态":
                disease_advice = f"检测到{recognition_result['analysis_result']['disease_name']}，" \
                               f"建议：{recognition_result['analysis_result']['treatment_advice']}"
                
                # 保存识别结果到决策记录
                try:
                    save_decision_record(
                        crop_type=crop_type,
                        growth_stage="图像识别",
                        advice=disease_advice,
                        field_id=field_id
                    )
                except Exception as e:
                    print(f"保存识别结果失败: {e}")

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], port=5000, host='0.0.0.0')

