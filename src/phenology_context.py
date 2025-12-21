#!/usr/bin/env python3
"""
物候数据上下文模块
实现ChinaCropPhen1km物候数据的加载和地理位置到物候期的映射
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import logging
from datetime import datetime, date

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import xy
    import pandas as pd
    RASTERIO_AVAILABLE = True
except ImportError:
    print("⚠️  rasterio未安装，物候数据功能将受限")
    RASTERIO_AVAILABLE = False

from src.dataset_config import get_dataset_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChinaCropPhenologyMapping:
    """中国作物物候期映射管理"""
    
    def __init__(self):
        """初始化物候期映射"""
        # 作物物候期定义（天数，从播种开始计算）
        self.crop_phenology_stages = {
            '水稻': {
                'stages': ['播种期', '出苗期', '分蘖期', '拔节期', '抽穗期', '灌浆期', '成熟期'],
                'days_from_sowing': [0, 15, 30, 60, 90, 120, 150],
                'growth_duration': 150,  # 总生长期天数
                'optimal_temp_range': (20, 35),  # 最适温度范围
                'critical_stages': ['抽穗期', '灌浆期']  # 关键生长期
            },
            '小麦': {
                'stages': ['播种期', '出苗期', '分蘖期', '拔节期', '抽穗期', '灌浆期', '成熟期'],
                'days_from_sowing': [0, 10, 25, 50, 80, 110, 140],
                'growth_duration': 140,
                'optimal_temp_range': (15, 25),
                'critical_stages': ['拔节期', '抽穗期']
            },
            '玉米': {
                'stages': ['播种期', '出苗期', '拔节期', '抽雄期', '灌浆期', '成熟期'],
                'days_from_sowing': [0, 8, 35, 65, 85, 120],
                'growth_duration': 120,
                'optimal_temp_range': (20, 30),
                'critical_stages': ['抽雄期', '灌浆期']
            },
            '大豆': {
                'stages': ['播种期', '出苗期', '分枝期', '开花期', '结荚期', '鼓粒期', '成熟期'],
                'days_from_sowing': [0, 7, 25, 45, 65, 85, 110],
                'growth_duration': 110,
                'optimal_temp_range': (18, 28),
                'critical_stages': ['开花期', '结荚期']
            },
            '马铃薯': {
                'stages': ['播种期', '出苗期', '块茎形成期', '块茎增长期', '淀粉积累期', '成熟期'],
                'days_from_sowing': [0, 15, 35, 55, 75, 100],
                'growth_duration': 100,
                'optimal_temp_range': (15, 25),
                'critical_stages': ['块茎形成期', '块茎增长期']
            },
            '番茄': {
                'stages': ['播种期', '出苗期', '花芽分化期', '开花期', '结果期', '成熟期'],
                'days_from_sowing': [0, 10, 30, 50, 70, 100],
                'growth_duration': 100,
                'optimal_temp_range': (18, 28),
                'critical_stages': ['开花期', '结果期']
            },
            '苹果': {
                'stages': ['萌芽期', '展叶期', '开花期', '幼果期', '果实膨大期', '成熟期'],
                'days_from_spring': [0, 20, 40, 60, 100, 150],  # 从春季开始计算
                'growth_duration': 200,  # 年生长周期
                'optimal_temp_range': (12, 25),
                'critical_stages': ['开花期', '果实膨大期']
            }
        }
        
        # 月份到物候期的映射（北半球）
        self.month_to_season = {
            1: '冬季', 2: '冬季', 3: '春季',
            4: '春季', 5: '春季', 6: '夏季',
            7: '夏季', 8: '夏季', 9: '秋季',
            10: '秋季', 11: '秋季', 12: '冬季'
        }
        
        # 季节对应的主要农事活动
        self.season_activities = {
            '春季': ['播种', '施肥', '灌溉', '病虫害防治'],
            '夏季': ['田间管理', '病虫害防治', '灌溉', '除草'],
            '秋季': ['收获', '储存', '土壤处理'],
            '冬季': ['休耕', '设施维护', '规划']
        }
        
        # 中国主要农业区域划分
        self.agricultural_regions = {
            '东北平原': {
                'provinces': ['黑龙江', '吉林', '辽宁'],
                'main_crops': ['玉米', '大豆', '水稻'],
                'climate_type': '温带大陆性',
                'growing_season': (4, 10)  # 4月到10月
            },
            '华北平原': {
                'provinces': ['河北', '山东', '河南', '北京', '天津'],
                'main_crops': ['小麦', '玉米', '棉花'],
                'climate_type': '温带季风',
                'growing_season': (3, 11)
            },
            '长江中下游平原': {
                'provinces': ['江苏', '安徽', '湖北', '湖南', '江西'],
                'main_crops': ['水稻', '小麦', '油菜'],
                'climate_type': '亚热带季风',
                'growing_season': (3, 11)
            },
            '华南地区': {
                'provinces': ['广东', '广西', '福建', '海南'],
                'main_crops': ['水稻', '甘蔗', '热带水果'],
                'climate_type': '热带亚热带季风',
                'growing_season': (1, 12)  # 全年
            },
            '西南地区': {
                'provinces': ['四川', '重庆', '云南', '贵州'],
                'main_crops': ['水稻', '玉米', '马铃薯'],
                'climate_type': '亚热带高原',
                'growing_season': (3, 11)
            },
            '西北地区': {
                'provinces': ['新疆', '甘肃', '宁夏', '青海', '陕西'],
                'main_crops': ['小麦', '玉米', '棉花'],
                'climate_type': '温带大陆性',
                'growing_season': (4, 10)
            }
        }
    
    def get_crop_phenology(self, crop_name: str) -> Optional[Dict[str, Any]]:
        """获取作物物候期信息"""
        return self.crop_phenology_stages.get(crop_name)
    
    def get_current_phenology_stage(self, crop_name: str, current_date: Union[date, datetime], sowing_date: Union[date, datetime]) -> Optional[str]:
        """根据当前日期和播种日期确定物候期"""
        crop_info = self.get_crop_phenology(crop_name)
        if not crop_info:
            return None
        
        if isinstance(current_date, datetime):
            current_date = current_date.date()
        if isinstance(sowing_date, datetime):
            sowing_date = sowing_date.date()
        
        days_since_sowing = (current_date - sowing_date).days
        
        stages = crop_info['stages']
        stage_days = crop_info['days_from_sowing']
        
        # 找到当前所处的物候期
        current_stage = stages[0]  # 默认为第一个阶段
        for i, stage_day in enumerate(stage_days):
            if days_since_sowing >= stage_day:
                current_stage = stages[i]
            else:
                break
        
        return current_stage
    
    def get_seasonal_context(self, month: int) -> Dict[str, Any]:
        """获取季节性上下文信息"""
        season = self.month_to_season.get(month, '未知')
        activities = self.season_activities.get(season, [])
        
        return {
            'season': season,
            'month': month,
            'main_activities': activities,
            'is_growing_season': month in range(3, 12)  # 3-11月为主要生长季节
        }
    
    def get_regional_context(self, province: str) -> Optional[Dict[str, Any]]:
        """根据省份获取区域农业上下文"""
        for region_name, region_info in self.agricultural_regions.items():
            if province in region_info['provinces']:
                return {
                    'region': region_name,
                    'climate_type': region_info['climate_type'],
                    'main_crops': region_info['main_crops'],
                    'growing_season_start': region_info['growing_season'][0],
                    'growing_season_end': region_info['growing_season'][1]
                }
        return None

class PhenologyDataLoader:
    """物候数据加载器"""
    
    def __init__(self):
        """初始化物候数据加载器"""
        self.config = get_dataset_config()
        self.phenology_config = self.config.phenology_dataset
        self.phenology_mapping = ChinaCropPhenologyMapping()
        
        # 数据缓存
        self.data_cache = {}
        self.metadata_cache = {}
        
        logger.info("物候数据加载器初始化完成")
    
    def analyze_phenology_dataset(self) -> Dict[str, Any]:
        """分析物候数据集结构"""
        if not os.path.exists(self.phenology_config.path):
            raise FileNotFoundError(f"物候数据集路径不存在: {self.phenology_config.path}")
        
        logger.info("开始分析物候数据集...")
        
        # 获取所有数据文件
        data_files = []
        file_types = defaultdict(int)
        total_size = 0
        
        for root, dirs, files in os.walk(self.phenology_config.path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in self.phenology_config.data_extensions:
                    data_files.append(file_path)
                    file_types[file_ext] += 1
                    
                    try:
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                    except OSError:
                        pass
        
        # 分析文件名模式以推断数据内容
        file_patterns = self._analyze_file_patterns(data_files)
        
        dataset_info = {
            'dataset_path': self.phenology_config.path,
            'total_files': len(data_files),
            'file_types': dict(file_types),
            'total_size_gb': total_size / (1024**3),
            'temporal_range': self.phenology_config.temporal_range,
            'spatial_resolution': self.phenology_config.spatial_resolution,
            'file_patterns': file_patterns,
            'sample_files': data_files[:10]  # 前10个文件作为样本
        }
        
        logger.info(f"物候数据集分析完成: {len(data_files)} 个文件, {total_size/(1024**3):.2f} GB")
        
        return dataset_info
    
    def _analyze_file_patterns(self, file_paths: List[str]) -> Dict[str, Any]:
        """分析文件名模式"""
        patterns = {
            'years': set(),
            'crops': set(),
            'phenology_stages': set(),
            'file_naming_pattern': None
        }
        
        # 从文件名中提取信息
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            
            # 提取年份
            import re
            year_matches = re.findall(r'(19|20)\d{2}', filename)
            patterns['years'].update(year_matches)
            
            # 检查是否包含作物名称
            for crop in self.phenology_mapping.crop_phenology_stages.keys():
                if crop in filename:
                    patterns['crops'].add(crop)
            
            # 检查物候期关键词
            phenology_keywords = ['播种', '出苗', '开花', '成熟', 'sowing', 'emergence', 'flowering', 'maturity']
            for keyword in phenology_keywords:
                if keyword in filename.lower():
                    patterns['phenology_stages'].add(keyword)
        
        # 转换为列表以便JSON序列化
        patterns['years'] = sorted(list(patterns['years']))
        patterns['crops'] = list(patterns['crops'])
        patterns['phenology_stages'] = list(patterns['phenology_stages'])
        
        return patterns
    
    def load_phenology_data(self, file_path: str, cache: bool = True) -> Optional[Dict[str, Any]]:
        """加载单个物候数据文件"""
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio未安装，无法加载栅格数据")
            return None
        
        if cache and file_path in self.data_cache:
            return self.data_cache[file_path]
        
        try:
            with rasterio.open(file_path) as dataset:
                # 读取元数据
                metadata = {
                    'width': dataset.width,
                    'height': dataset.height,
                    'count': dataset.count,
                    'dtype': str(dataset.dtypes[0]),
                    'crs': str(dataset.crs),
                    'transform': dataset.transform,
                    'bounds': dataset.bounds,
                    'nodata': dataset.nodata
                }
                
                # 读取数据（如果文件不太大）
                data = None
                if dataset.width * dataset.height < 10000:  # 小于10K像素
                    data = dataset.read(1)  # 读取第一个波段
                
                result = {
                    'metadata': metadata,
                    'data': data,
                    'file_path': file_path
                }
                
                if cache:
                    self.data_cache[file_path] = result
                
                return result
                
        except Exception as e:
            logger.error(f"无法加载物候数据文件 {file_path}: {e}")
            return None
    
    def get_phenology_context(self, 
                            latitude: float, 
                            longitude: float, 
                            crop_type: str,
                            current_date: Optional[Union[date, datetime]] = None) -> Dict[str, Any]:
        """
        获取指定地理位置和作物的物候上下文信息
        
        Args:
            latitude: 纬度
            longitude: 经度
            crop_type: 作物类型
            current_date: 当前日期，如果为None则使用今天
            
        Returns:
            物候上下文信息字典
        """
        if current_date is None:
            current_date = datetime.now().date()
        elif isinstance(current_date, datetime):
            current_date = current_date.date()
        
        # 获取作物物候期信息
        crop_phenology = self.phenology_mapping.get_crop_phenology(crop_type)
        
        # 获取季节性上下文
        seasonal_context = self.phenology_mapping.get_seasonal_context(current_date.month)
        
        # 根据地理位置推断区域（简化实现）
        region_context = self._infer_region_from_coordinates(latitude, longitude)
        
        # 构建综合上下文
        context = {
            'location': {
                'latitude': latitude,
                'longitude': longitude,
                'region': region_context
            },
            'crop_info': {
                'crop_type': crop_type,
                'phenology_stages': crop_phenology['stages'] if crop_phenology else [],
                'optimal_temp_range': crop_phenology['optimal_temp_range'] if crop_phenology else None,
                'critical_stages': crop_phenology['critical_stages'] if crop_phenology else []
            },
            'temporal_context': {
                'current_date': current_date.isoformat(),
                'season': seasonal_context['season'],
                'month': seasonal_context['month'],
                'is_growing_season': seasonal_context['is_growing_season'],
                'main_activities': seasonal_context['main_activities']
            },
            'recommendations': self._generate_phenology_recommendations(
                crop_type, seasonal_context, region_context
            )
        }
        
        return context
    
    def _infer_region_from_coordinates(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """根据经纬度推断农业区域（简化实现）"""
        # 中国大致的地理范围
        if not (18 <= latitude <= 54 and 73 <= longitude <= 135):
            return None
        
        # 简化的区域划分（基于经纬度范围）
        if latitude >= 45:  # 东北地区
            return self.phenology_mapping.get_regional_context('黑龙江')
        elif latitude >= 35 and longitude <= 110:  # 华北地区
            return self.phenology_mapping.get_regional_context('河北')
        elif 28 <= latitude < 35:  # 长江中下游
            return self.phenology_mapping.get_regional_context('江苏')
        elif latitude < 28 and longitude > 110:  # 华南地区
            return self.phenology_mapping.get_regional_context('广东')
        elif longitude <= 100:  # 西北地区
            return self.phenology_mapping.get_regional_context('新疆')
        else:  # 西南地区
            return self.phenology_mapping.get_regional_context('四川')
    
    def _generate_phenology_recommendations(self, 
                                          crop_type: str, 
                                          seasonal_context: Dict[str, Any],
                                          region_context: Optional[Dict[str, Any]]) -> List[str]:
        """生成基于物候期的农事建议"""
        recommendations = []
        
        season = seasonal_context['season']
        activities = seasonal_context['main_activities']
        
        # 基于季节的通用建议
        for activity in activities:
            recommendations.append(f"当前{season}，建议进行{activity}")
        
        # 基于作物类型的特定建议
        crop_info = self.phenology_mapping.get_crop_phenology(crop_type)
        if crop_info:
            critical_stages = crop_info.get('critical_stages', [])
            if critical_stages:
                recommendations.append(f"{crop_type}的关键生长期为{', '.join(critical_stages)}，需要特别关注")
        
        # 基于区域的建议
        if region_context:
            region_crops = region_context.get('main_crops', [])
            if crop_type in region_crops:
                recommendations.append(f"{crop_type}是{region_context['region']}的主要作物，适合当地种植")
        
        return recommendations

# 便捷函数
def create_phenology_loader() -> PhenologyDataLoader:
    """创建物候数据加载器"""
    return PhenologyDataLoader()

def get_phenology_mapping() -> ChinaCropPhenologyMapping:
    """获取物候期映射"""
    return ChinaCropPhenologyMapping()

if __name__ == "__main__":
    # 测试物候数据模块
    print("🧪 物候数据上下文模块测试")
    print("=" * 60)
    
    try:
        # 测试物候期映射
        print("📋 测试物候期映射...")
        mapping = get_phenology_mapping()
        
        # 测试作物物候期信息
        rice_phenology = mapping.get_crop_phenology('水稻')
        print(f"✅ 水稻物候期信息:")
        print(f"   生长阶段: {rice_phenology['stages']}")
        print(f"   生长周期: {rice_phenology['growth_duration']} 天")
        print(f"   关键期: {rice_phenology['critical_stages']}")
        
        # 测试季节性上下文
        seasonal = mapping.get_seasonal_context(6)  # 6月
        print(f"✅ 6月季节性上下文:")
        print(f"   季节: {seasonal['season']}")
        print(f"   主要活动: {seasonal['main_activities']}")
        
        # 测试区域上下文
        regional = mapping.get_regional_context('江苏')
        print(f"✅ 江苏区域上下文:")
        print(f"   农业区: {regional['region']}")
        print(f"   主要作物: {regional['main_crops']}")
        
        # 测试物候数据加载器
        print(f"\n🔍 测试物候数据加载器...")
        loader = create_phenology_loader()
        
        # 分析数据集
        dataset_info = loader.analyze_phenology_dataset()
        print(f"✅ 物候数据集分析完成:")
        print(f"   总文件数: {dataset_info['total_files']}")
        print(f"   数据大小: {dataset_info['total_size_gb']:.2f} GB")
        print(f"   时间范围: {dataset_info['temporal_range']}")
        print(f"   发现的年份: {dataset_info['file_patterns']['years'][:5]}")
        
        # 测试物候上下文获取
        print(f"\n📊 测试物候上下文获取...")
        context = loader.get_phenology_context(
            latitude=32.0,  # 南京附近
            longitude=118.8,
            crop_type='水稻',
            current_date=datetime(2024, 6, 15)
        )
        
        print(f"✅ 物候上下文获取完成:")
        print(f"   地理位置: {context['location']['latitude']}, {context['location']['longitude']}")
        print(f"   作物类型: {context['crop_info']['crop_type']}")
        print(f"   当前季节: {context['temporal_context']['season']}")
        print(f"   农事建议: {context['recommendations'][:3]}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 物候数据上下文模块测试完成")