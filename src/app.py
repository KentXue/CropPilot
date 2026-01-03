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

        # 尝试进行AI图像识别
        recognition_result = None
        try:
            from image_recognition import analyze_crop_image
            print(f"开始AI图像识别: {saved_path}, 作物类型: {crop_type}")
            recognition_result = analyze_crop_image(saved_path, crop_type)
            print(f"AI识别结果: {recognition_result}")
        except ImportError as e:
            print(f"图像识别模块导入失败: {e}")
        except Exception as e:
            print(f"AI图像识别失败: {e}")

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
            "image_path": f"/{relative_path}",
            "crop_type": crop_type
        }

        # 如果AI图像识别成功，添加识别结果
        if recognition_result and recognition_result.get('status') == 'success':
            response_data["recognition"] = recognition_result["analysis_result"]
            response_data["recognition_method"] = recognition_result.get("method", "unknown")
            
            # 获取主要识别结果
            primary_result = recognition_result["analysis_result"].get("primary_result", {})
            disease_name = primary_result.get("disease_name", "未知")
            confidence = primary_result.get("confidence", 0)
            treatment = primary_result.get("treatment_advice", "请咨询专家")
            
            # 如果识别出病害且置信度足够高，自动生成防治建议
            if disease_name not in ["健康状态", "健康", "healthy"] and confidence > 0.5:
                disease_advice = f"AI识别结果：{disease_name}（置信度：{confidence:.2%}）。{treatment}"
                
                # 保存AI识别结果到决策记录
                try:
                    save_decision_record(
                        crop_type=crop_type,
                        growth_stage="AI图像识别",
                        advice=disease_advice,
                        field_id=field_id
                    )
                    response_data["auto_advice_saved"] = True
                except Exception as e:
                    print(f"保存AI识别结果失败: {e}")
                    response_data["auto_advice_saved"] = False
            
            # 添加识别摘要信息
            response_data["recognition_summary"] = {
                "disease_detected": disease_name,
                "confidence": confidence,
                "is_healthy": disease_name in ["健康状态", "健康", "healthy"],
                "method_used": recognition_result.get("method", "unknown")
            }
        else:
            # AI识别失败时的处理
            if recognition_result:
                response_data["recognition_error"] = recognition_result.get("message", "识别失败")
            else:
                response_data["recognition_error"] = "AI识别模块不可用"

        return jsonify(response_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/check_field_alerts', methods=['GET'])
def api_check_field_alerts():
    """检查地块预警情况"""
    try:
        field_id = request.args.get('field_id', type=int)
        if not field_id:
            return jsonify({
                "status": "error",
                "message": "缺少field_id参数"
            }), 400
        
        # 导入异常检测模块
        from anomaly_detection import get_anomaly_detector
        
        detector = get_anomaly_detector()
        result = detector.check_field_alerts(field_id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/check_sensor_anomalies', methods=['POST'])
def api_check_sensor_anomalies():
    """检查传感器数据异常"""
    try:
        data = request.json
        sensor_data = data.get('sensor_data', {})
        crop_type = data.get('crop_type', '水稻')
        
        if not sensor_data:
            return jsonify({
                "status": "error",
                "message": "缺少sensor_data参数"
            }), 400
        
        # 导入异常检测模块
        from anomaly_detection import get_anomaly_detector
        
        detector = get_anomaly_detector()
        anomalies = detector.detect_sensor_data_anomalies(sensor_data, crop_type)
        alert_summary = detector.generate_alert_summary(anomalies)
        
        return jsonify({
            "status": "success",
            "anomalies": anomalies,
            "alert_summary": alert_summary
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/generate_demo_data', methods=['POST'])
def api_generate_demo_data():
    """生成演示用的传感器数据"""
    try:
        data = request.json
        field_id = data.get('field_id')
        days = data.get('days', 7)  # 默认生成7天的数据
        
        if not field_id:
            return jsonify({
                "status": "error",
                "message": "缺少field_id参数"
            }), 400
        
        # 获取地块信息
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT crop_type FROM fields WHERE id = %s", (field_id,))
                field = cursor.fetchone()
                if not field:
                    return jsonify({
                        "status": "error",
                        "message": "地块不存在"
                    }), 404
                
                crop_type = field.get('crop_type', '水稻')
                
                # 生成模拟数据
                import random
                from datetime import datetime, timedelta
                
                base_time = datetime.now() - timedelta(days=days)
                records_created = 0
                
                for i in range(days * 4):  # 每天4条记录
                    record_time = base_time + timedelta(hours=i * 6)  # 每6小时一条
                    
                    # 根据作物类型生成不同的数据范围
                    if crop_type == '水稻':
                        temp_base, humidity_base = 28, 75
                        soil_moisture_base, ph_base = 80, 6.5
                    else:  # 玉米等
                        temp_base, humidity_base = 25, 65
                        soil_moisture_base, ph_base = 70, 6.8
                    
                    # 添加随机波动
                    temperature = round(temp_base + random.uniform(-5, 8), 1)
                    humidity = round(humidity_base + random.uniform(-15, 20), 1)
                    soil_moisture = round(soil_moisture_base + random.uniform(-20, 15), 1)
                    ph_value = round(ph_base + random.uniform(-0.8, 1.2), 1)
                    light_intensity = random.randint(20000, 80000)
                    nitrogen = round(random.uniform(100, 200), 1)
                    phosphorus = round(random.uniform(50, 120), 1)
                    potassium = round(random.uniform(80, 150), 1)
                    
                    # 插入数据
                    sql = """
                        INSERT INTO sensor_data 
                        (field_id, crop_type, growth_stage, temperature, humidity, 
                         soil_moisture, light_intensity, ph_value, nitrogen, 
                         phosphorus, potassium, location, recorded_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(sql, (
                        field_id, crop_type, '分蘖期', temperature, humidity,
                        soil_moisture, light_intensity, ph_value, nitrogen,
                        phosphorus, potassium, f'演示数据-{field_id}', record_time
                    ))
                    records_created += 1
                
                conn.commit()
                
                return jsonify({
                    "status": "success",
                    "message": f"成功生成 {records_created} 条演示数据",
                    "records_created": records_created
                })
                
        finally:
            conn.close()
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/analyze_image', methods=['POST'])
def api_analyze_image():
    """专门的图像识别API端点"""
    try:
        # 检查是否有上传的文件
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "请上传图片文件"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "文件名不能为空"}), 400

        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "不支持的文件类型"}), 400

        # 获取可选参数
        crop_type = request.form.get('crop_type', '')
        field_id = request.form.get('field_id', type=int)

        # 如果提供了field_id，尝试获取作物类型
        if field_id and not crop_type:
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

        # 保存临时文件
        upload_folder = app.config.get('UPLOAD_FOLDER')
        os.makedirs(upload_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        safe_name = secure_filename(file.filename)
        temp_filename = f"temp_{timestamp}_{safe_name}"
        temp_path = os.path.join(upload_folder, temp_filename)
        file.save(temp_path)

        try:
            # 进行AI图像识别
            from image_recognition import analyze_crop_image
            recognition_result = analyze_crop_image(temp_path, crop_type)
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if recognition_result.get('status') == 'success':
                return jsonify({
                    "status": "success",
                    "crop_type": crop_type,
                    "recognition_result": recognition_result["analysis_result"],
                    "method": recognition_result.get("method", "unknown"),
                    "image_info": recognition_result.get("image_info", {})
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": recognition_result.get("message", "识别失败")
                }), 500
                
        except ImportError:
            return jsonify({
                "status": "error",
                "message": "AI图像识别模块不可用，请安装相关依赖"
            }), 500
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                "status": "error",
                "message": f"图像识别失败: {str(e)}"
            }), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/get_supported_diseases', methods=['GET'])
def api_get_supported_diseases():
    """获取支持识别的病害列表"""
    try:
        from image_recognition import get_plant_classifier
        classifier = get_plant_classifier()
        
        if classifier.available:
            diseases = classifier.get_supported_diseases()
            return jsonify({
                "status": "success",
                "supported_diseases": diseases,
                "total_count": len(diseases),
                "ai_available": True
            })
        else:
            return jsonify({
                "status": "success",
                "supported_diseases": ["基础规则识别"],
                "total_count": 1,
                "ai_available": False,
                "message": "AI模块不可用，使用基础识别"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    print("🌾 CropPilot 智能农业决策支持系统启动中...")
    print("📊 功能模块:")
    print("  ✅ 基础决策引擎")
    print("  ✅ 智能知识库检索")
    print("  ✅ 传感器数据管理")
    print("  ✅ 数据可视化")
    print("  ✅ 异常检测预警")
    
    # 检查AI图像识别模块
    try:
        from image_recognition import get_plant_classifier
        classifier = get_plant_classifier()
        if classifier.available:
            print("  ✅ AI图像识别 (深度学习)")
            print(f"     - 设备: {classifier.device}")
            print(f"     - 支持病害: {len(classifier.class_names)}种")
        else:
            print("  ⚠️  AI图像识别 (基础规则)")
            print("     - 提示: 安装PyTorch获得完整AI功能")
    except Exception as e:
        print(f"  ❌ AI图像识别模块加载失败: {e}")
    
    print(f"\n🚀 服务启动: http://localhost:5000")
    app.run(debug=app.config['DEBUG'], port=5000, host='0.0.0.0')

