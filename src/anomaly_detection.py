#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测与预警模块
用于监测传感器数据异常并生成预警
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class AnomalyDetector:
    """传感器数据异常检测器"""
    
    def __init__(self):
        # 定义各作物的正常参数范围
        self.normal_ranges = {
            "水稻": {
                "temperature": {"min": 15, "max": 35, "optimal_min": 20, "optimal_max": 30},
                "humidity": {"min": 50, "max": 95, "optimal_min": 60, "optimal_max": 85},
                "soil_moisture": {"min": 60, "max": 95, "optimal_min": 70, "optimal_max": 90},
                "ph_value": {"min": 5.5, "max": 7.5, "optimal_min": 6.0, "optimal_max": 7.0},
                "light_intensity": {"min": 10000, "max": 100000, "optimal_min": 30000, "optimal_max": 80000},
                "nitrogen": {"min": 80, "max": 250, "optimal_min": 120, "optimal_max": 200},
                "phosphorus": {"min": 30, "max": 150, "optimal_min": 50, "optimal_max": 120},
                "potassium": {"min": 60, "max": 200, "optimal_min": 80, "optimal_max": 160}
            },
            "玉米": {
                "temperature": {"min": 10, "max": 40, "optimal_min": 18, "optimal_max": 32},
                "humidity": {"min": 40, "max": 85, "optimal_min": 50, "optimal_max": 75},
                "soil_moisture": {"min": 50, "max": 85, "optimal_min": 60, "optimal_max": 80},
                "ph_value": {"min": 6.0, "max": 8.0, "optimal_min": 6.5, "optimal_max": 7.5},
                "light_intensity": {"min": 15000, "max": 120000, "optimal_min": 40000, "optimal_max": 100000},
                "nitrogen": {"min": 100, "max": 300, "optimal_min": 150, "optimal_max": 250},
                "phosphorus": {"min": 40, "max": 180, "optimal_min": 60, "optimal_max": 150},
                "potassium": {"min": 80, "max": 250, "optimal_min": 100, "optimal_max": 200}
            },
            "小麦": {
                "temperature": {"min": 5, "max": 30, "optimal_min": 12, "optimal_max": 25},
                "humidity": {"min": 45, "max": 80, "optimal_min": 55, "optimal_max": 70},
                "soil_moisture": {"min": 55, "max": 80, "optimal_min": 60, "optimal_max": 75},
                "ph_value": {"min": 6.0, "max": 8.0, "optimal_min": 6.5, "optimal_max": 7.5},
                "light_intensity": {"min": 20000, "max": 100000, "optimal_min": 35000, "optimal_max": 85000},
                "nitrogen": {"min": 90, "max": 220, "optimal_min": 120, "optimal_max": 180},
                "phosphorus": {"min": 35, "max": 140, "optimal_min": 50, "optimal_max": 110},
                "potassium": {"min": 70, "max": 180, "optimal_min": 90, "optimal_max": 150}
            }
        }
        
        # 异常等级定义
        self.alert_levels = {
            "info": {"color": "#17a2b8", "icon": "ℹ️", "priority": 1},
            "warning": {"color": "#ffc107", "icon": "⚠️", "priority": 2},
            "danger": {"color": "#dc3545", "icon": "🚨", "priority": 3},
            "critical": {"color": "#6f42c1", "icon": "💀", "priority": 4}
        }
        
        # 预警消息模板
        self.alert_messages = {
            "temperature": {
                "too_low": "温度过低，可能影响作物生长，建议采取保温措施",
                "too_high": "温度过高，可能导致作物热害，建议降温或遮阳",
                "optimal": "温度适宜，有利于作物生长"
            },
            "humidity": {
                "too_low": "湿度过低，可能导致作物缺水，建议增加灌溉或喷雾",
                "too_high": "湿度过高，容易滋生病害，建议加强通风",
                "optimal": "湿度适宜，环境条件良好"
            },
            "soil_moisture": {
                "too_low": "土壤湿度不足，作物可能缺水，建议及时灌溉",
                "too_high": "土壤过湿，可能导致根系缺氧，建议排水",
                "optimal": "土壤湿度适宜，有利于根系发育"
            },
            "ph_value": {
                "too_low": "土壤偏酸，可能影响养分吸收，建议施用石灰调节",
                "too_high": "土壤偏碱，可能影响微量元素吸收，建议施用硫磺调节",
                "optimal": "土壤pH值适宜，有利于养分吸收"
            },
            "light_intensity": {
                "too_low": "光照不足，可能影响光合作用，建议补光或调整种植密度",
                "too_high": "光照过强，可能导致叶片灼伤，建议遮阳",
                "optimal": "光照充足，有利于光合作用"
            },
            "nitrogen": {
                "too_low": "氮素不足，叶片可能发黄，建议追施氮肥",
                "too_high": "氮素过量，可能导致徒长，建议控制氮肥用量",
                "optimal": "氮素含量适宜，有利于茎叶生长"
            },
            "phosphorus": {
                "too_low": "磷素不足，可能影响根系发育，建议施用磷肥",
                "too_high": "磷素过量，可能影响其他元素吸收，建议平衡施肥",
                "optimal": "磷素含量适宜，有利于根系和花果发育"
            },
            "potassium": {
                "too_low": "钾素不足，可能影响抗逆性，建议施用钾肥",
                "too_high": "钾素过量，可能影响钙镁吸收，建议平衡施肥",
                "optimal": "钾素含量适宜，有利于提高抗逆性"
            }
        }
    
    def detect_single_value_anomaly(self, parameter: str, value: float, crop_type: str) -> Dict[str, Any]:
        """检测单个参数值的异常"""
        if crop_type not in self.normal_ranges:
            crop_type = "水稻"  # 默认使用水稻标准
        
        if parameter not in self.normal_ranges[crop_type]:
            return {"status": "unknown", "message": "未知参数类型"}
        
        ranges = self.normal_ranges[crop_type][parameter]
        messages = self.alert_messages.get(parameter, {})
        
        # 判断异常等级
        if value < ranges["min"]:
            level = "critical" if value < ranges["min"] * 0.8 else "danger"
            status = "too_low"
        elif value > ranges["max"]:
            level = "critical" if value > ranges["max"] * 1.2 else "danger"
            status = "too_high"
        elif value < ranges["optimal_min"]:
            level = "warning"
            status = "too_low"
        elif value > ranges["optimal_max"]:
            level = "warning"
            status = "too_high"
        else:
            level = "info"
            status = "optimal"
        
        return {
            "parameter": parameter,
            "value": value,
            "status": status,
            "level": level,
            "message": messages.get(status, f"{parameter}数值异常"),
            "ranges": ranges,
            "alert_info": self.alert_levels[level]
        }
    
    def detect_sensor_data_anomalies(self, sensor_data: Dict[str, Any], crop_type: str = "水稻") -> List[Dict[str, Any]]:
        """检测传感器数据中的所有异常"""
        anomalies = []
        
        # 检测的参数列表
        parameters_to_check = [
            "temperature", "humidity", "soil_moisture", "ph_value",
            "light_intensity", "nitrogen", "phosphorus", "potassium"
        ]
        
        for param in parameters_to_check:
            value = sensor_data.get(param)
            if value is not None:
                try:
                    value = float(value)
                    anomaly = self.detect_single_value_anomaly(param, value, crop_type)
                    if anomaly["level"] in ["warning", "danger", "critical"]:
                        anomalies.append(anomaly)
                except (ValueError, TypeError):
                    continue
        
        # 按优先级排序
        anomalies.sort(key=lambda x: self.alert_levels[x["level"]]["priority"], reverse=True)
        
        return anomalies
    
    def detect_trend_anomalies(self, sensor_data_list: List[Dict[str, Any]], crop_type: str = "水稻") -> List[Dict[str, Any]]:
        """检测数据趋势异常"""
        if len(sensor_data_list) < 3:
            return []
        
        trend_anomalies = []
        
        # 检测急剧变化
        parameters = ["temperature", "humidity", "soil_moisture", "ph_value"]
        
        for param in parameters:
            values = []
            for data in sensor_data_list[-5:]:  # 取最近5条数据
                if data.get(param) is not None:
                    try:
                        values.append(float(data[param]))
                    except (ValueError, TypeError):
                        continue
            
            if len(values) >= 3:
                # 检测急剧变化
                recent_change = abs(values[-1] - values[-2]) if len(values) >= 2 else 0
                avg_change = sum(abs(values[i] - values[i-1]) for i in range(1, len(values))) / (len(values) - 1) if len(values) > 1 else 0
                
                # 如果最近变化超过平均变化的3倍，认为是异常
                if recent_change > avg_change * 3 and avg_change > 0:
                    trend_anomalies.append({
                        "type": "trend",
                        "parameter": param,
                        "message": f"{param}数值变化异常，最近变化幅度: {recent_change:.2f}",
                        "level": "warning",
                        "recent_change": recent_change,
                        "avg_change": avg_change,
                        "alert_info": self.alert_levels["warning"]
                    })
        
        return trend_anomalies
    
    def generate_alert_summary(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成预警摘要"""
        if not anomalies:
            return {
                "total_alerts": 0,
                "max_level": "info",
                "summary": "所有参数正常",
                "recommendations": ["继续保持良好的田间管理"]
            }
        
        # 统计各等级的异常数量
        level_counts = {"info": 0, "warning": 0, "danger": 0, "critical": 0}
        for anomaly in anomalies:
            level = anomaly.get("level", "info")
            level_counts[level] += 1
        
        # 确定最高等级
        max_level = "info"
        for level in ["critical", "danger", "warning", "info"]:
            if level_counts[level] > 0:
                max_level = level
                break
        
        # 生成摘要
        total_alerts = sum(level_counts.values())
        
        if max_level == "critical":
            summary = f"发现{total_alerts}个异常，包含严重问题，需要立即处理"
        elif max_level == "danger":
            summary = f"发现{total_alerts}个异常，存在危险情况，建议尽快处理"
        elif max_level == "warning":
            summary = f"发现{total_alerts}个异常，需要关注并调整"
        else:
            summary = f"发现{total_alerts}个轻微异常，建议优化"
        
        # 生成建议
        recommendations = []
        for anomaly in anomalies[:3]:  # 取前3个最重要的异常
            recommendations.append(anomaly.get("message", ""))
        
        return {
            "total_alerts": total_alerts,
            "level_counts": level_counts,
            "max_level": max_level,
            "summary": summary,
            "recommendations": recommendations,
            "alert_info": self.alert_levels[max_level]
        }
    
    def check_field_alerts(self, field_id: int, crop_type: str = None) -> Dict[str, Any]:
        """检查指定地块的预警情况"""
        try:
            from database import get_connection
            
            conn = get_connection()
            try:
                with conn.cursor() as cursor:
                    # 获取地块信息
                    if not crop_type:
                        cursor.execute("SELECT crop_type FROM fields WHERE id = %s", (field_id,))
                        field = cursor.fetchone()
                        crop_type = field.get('crop_type', '水稻') if field else '水稻'
                    
                    # 获取最近的传感器数据
                    cursor.execute("""
                        SELECT * FROM sensor_data 
                        WHERE field_id = %s 
                        ORDER BY recorded_at DESC 
                        LIMIT 10
                    """, (field_id,))
                    
                    sensor_data_list = cursor.fetchall()
                    
                    if not sensor_data_list:
                        return {
                            "field_id": field_id,
                            "crop_type": crop_type,
                            "status": "no_data",
                            "message": "该地块暂无传感器数据"
                        }
                    
                    # 检测最新数据的异常
                    latest_data = sensor_data_list[0]
                    current_anomalies = self.detect_sensor_data_anomalies(latest_data, crop_type)
                    
                    # 检测趋势异常
                    trend_anomalies = self.detect_trend_anomalies(sensor_data_list, crop_type)
                    
                    # 合并所有异常
                    all_anomalies = current_anomalies + trend_anomalies
                    
                    # 生成摘要
                    alert_summary = self.generate_alert_summary(all_anomalies)
                    
                    return {
                        "field_id": field_id,
                        "crop_type": crop_type,
                        "status": "success",
                        "latest_data_time": latest_data.get('recorded_at'),
                        "current_anomalies": current_anomalies,
                        "trend_anomalies": trend_anomalies,
                        "all_anomalies": all_anomalies,
                        "alert_summary": alert_summary
                    }
                    
            finally:
                conn.close()
                
        except Exception as e:
            return {
                "field_id": field_id,
                "status": "error",
                "message": f"检查预警失败: {str(e)}"
            }

# 全局实例
anomaly_detector = None

def get_anomaly_detector():
    """获取异常检测器实例（单例模式）"""
    global anomaly_detector
    if anomaly_detector is None:
        anomaly_detector = AnomalyDetector()
    return anomaly_detector

def check_sensor_anomalies(sensor_data: Dict[str, Any], crop_type: str = "水稻") -> List[Dict[str, Any]]:
    """便捷的异常检测函数"""
    detector = get_anomaly_detector()
    return detector.detect_sensor_data_anomalies(sensor_data, crop_type)

if __name__ == "__main__":
    # 测试异常检测功能
    print("测试异常检测模块...")
    
    detector = AnomalyDetector()
    
    # 测试数据
    test_data = {
        "temperature": 45,  # 过高
        "humidity": 30,     # 过低
        "soil_moisture": 85, # 正常
        "ph_value": 6.5,    # 正常
        "nitrogen": 50      # 过低
    }
    
    anomalies = detector.detect_sensor_data_anomalies(test_data, "水稻")
    
    print(f"检测到 {len(anomalies)} 个异常:")
    for anomaly in anomalies:
        print(f"- {anomaly['parameter']}: {anomaly['message']} ({anomaly['level']})")
    
    summary = detector.generate_alert_summary(anomalies)
    print(f"\n预警摘要: {summary['summary']}")