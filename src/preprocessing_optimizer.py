#!/usr/bin/env python3
"""
图像预处理性能优化器
实现批量预处理、多线程数据加载和内存优化
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator
import logging
from dataclasses import dataclass
from queue import Queue
import gc

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    import cv2
    import torch
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    from PIL import Image
    import psutil
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

from src.image_preprocessing import PlantDiseasePreprocessor, PreprocessingMode
from src.data_augmentation import PlantDiseaseAugmentation, AugmentationStrategy

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    total_images: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    images_per_second: float = 0.0
    avg_image_size_mb: float = 0.0
    cache_hit_rate: float = 0.0

class ImageCache:
    """图像缓存管理器"""
    
    def __init__(self, max_size_mb: int = 1024):
        """
        初始化缓存
        
        Args:
            max_size_mb: 最大缓存大小（MB）
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.current_size = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """获取缓存的图像"""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key].copy()
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, image: np.ndarray) -> None:
        """添加图像到缓存"""
        with self._lock:
            image_size = image.nbytes
            
            # 如果图像太大，不缓存
            if image_size > self.max_size_bytes * 0.1:
                return
            
            # 清理空间
            while self.current_size + image_size > self.max_size_bytes and self.cache:
                self._evict_lru()
            
            # 添加到缓存
            self.cache[key] = image.copy()
            self.access_times[key] = time.time()
            self.current_size += image_size
    
    def _evict_lru(self) -> None:
        """移除最近最少使用的项目"""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        image_size = self.cache[lru_key].nbytes
        
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.current_size -= image_size
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.current_size = 0
            self.hits = 0
            self.misses = 0

class OptimizedImageDataset(Dataset):
    """优化的图像数据集"""
    
    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 preprocessor: PlantDiseasePreprocessor,
                 cache_enabled: bool = True,
                 prefetch_factor: int = 2):
        """
        初始化优化数据集
        
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            preprocessor: 预处理器
            cache_enabled: 是否启用缓存
            prefetch_factor: 预取因子
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.cache_enabled = cache_enabled
        self.prefetch_factor = prefetch_factor
        
        # 初始化缓存
        if cache_enabled:
            self.cache = ImageCache(max_size_mb=512)
        else:
            self.cache = None
        
        # 预加载统计信息
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集统计信息"""
        sample_size = min(100, len(self.image_paths))
        sample_paths = self.image_paths[:sample_size]
        
        total_size = 0
        valid_images = 0
        
        for path in sample_paths:
            try:
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    total_size += size
                    valid_images += 1
            except:
                continue
        
        self.avg_image_size = total_size / valid_images if valid_images > 0 else 0
        logger.info(f"数据集分析完成: 平均图像大小 {self.avg_image_size/1024/1024:.2f} MB")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 尝试从缓存获取
        image = None
        if self.cache_enabled and self.cache:
            image = self.cache.get(image_path)
        
        # 如果缓存未命中，加载图像
        if image is None:
            image = self._load_image(image_path)
            
            # 添加到缓存
            if self.cache_enabled and self.cache and image is not None:
                self.cache.put(image_path, image)
        
        # 预处理
        if image is not None:
            try:
                processed_image = self.preprocessor.preprocess_image(image)
                return processed_image, label
            except Exception as e:
                logger.warning(f"预处理失败 {image_path}: {e}")
                # 返回零张量作为备用
                return torch.zeros(3, 224, 224), label
        else:
            return torch.zeros(3, 224, 224), label
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载单张图像"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        except Exception as e:
            logger.warning(f"图像加载失败 {image_path}: {e}")
        return None

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, 
                 batch_size: int = 32,
                 num_workers: int = None,
                 use_multiprocessing: bool = False):
        """
        初始化批量处理器
        
        Args:
            batch_size: 批大小
            num_workers: 工作进程数
            use_multiprocessing: 是否使用多进程
        """
        self.batch_size = batch_size
        self.num_workers = num_workers or min(cpu_count(), 8)
        self.use_multiprocessing = use_multiprocessing
        
        logger.info(f"批量处理器初始化: batch_size={batch_size}, workers={self.num_workers}")
    
    def process_batch_parallel(self,
                             image_paths: List[str],
                             preprocessor: PlantDiseasePreprocessor) -> List[torch.Tensor]:
        """
        并行处理图像批次
        
        Args:
            image_paths: 图像路径列表
            preprocessor: 预处理器
            
        Returns:
            处理后的图像张量列表
        """
        if self.use_multiprocessing:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        
        with executor_class(max_workers=self.num_workers) as executor:
            # 提交任务
            futures = []
            for path in image_paths:
                future = executor.submit(self._process_single_image, path, preprocessor)
                futures.append(future)
            
            # 收集结果
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30秒超时
                    results.append(result)
                except Exception as e:
                    logger.warning(f"批量处理失败: {e}")
                    results.append(torch.zeros(3, 224, 224))
        
        return results
    
    def _process_single_image(self, 
                            image_path: str, 
                            preprocessor: PlantDiseasePreprocessor) -> torch.Tensor:
        """处理单张图像"""
        try:
            return preprocessor.preprocess_image(image_path)
        except Exception as e:
            logger.warning(f"图像处理失败 {image_path}: {e}")
            return torch.zeros(3, 224, 224)

class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        """初始化内存优化器"""
        self.initial_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def optimize_memory(self) -> Dict[str, float]:
        """执行内存优化"""
        before_memory = self._get_memory_usage()
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        after_memory = self._get_memory_usage()
        freed_memory = before_memory - after_memory
        
        return {
            'before_mb': before_memory,
            'after_mb': after_memory,
            'freed_mb': freed_memory
        }
    
    def monitor_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """监控函数执行时的内存使用"""
        start_memory = self._get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_memory = self._get_memory_usage()
        peak_memory = max(start_memory, end_memory)
        
        return result, {
            'start_mb': start_memory,
            'end_mb': end_memory,
            'peak_mb': peak_memory,
            'delta_mb': end_memory - start_memory
        }

class PreprocessingOptimizer:
    """预处理性能优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.memory_optimizer = MemoryOptimizer()
        self.performance_history = []
        
    def create_optimized_dataloader(self,
                                  image_paths: List[str],
                                  labels: List[int],
                                  preprocessor: PlantDiseasePreprocessor,
                                  batch_size: int = 32,
                                  num_workers: int = None,
                                  pin_memory: bool = None,
                                  prefetch_factor: int = 2) -> DataLoader:
        """
        创建优化的数据加载器
        
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            preprocessor: 预处理器
            batch_size: 批大小
            num_workers: 工作进程数
            pin_memory: 是否固定内存
            prefetch_factor: 预取因子
            
        Returns:
            优化的DataLoader
        """
        # 自动确定最优参数
        if num_workers is None:
            num_workers = min(cpu_count(), 8)
        
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        
        # 创建优化数据集
        dataset = OptimizedImageDataset(
            image_paths=image_paths,
            labels=labels,
            preprocessor=preprocessor,
            cache_enabled=True,
            prefetch_factor=prefetch_factor
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0,
            drop_last=False
        )
        
        logger.info(f"优化数据加载器创建完成: batch_size={batch_size}, workers={num_workers}")
        
        return dataloader
    
    def benchmark_preprocessing(self,
                              image_paths: List[str],
                              preprocessor: PlantDiseasePreprocessor,
                              batch_sizes: List[int] = None,
                              num_workers_list: List[int] = None) -> Dict[str, Any]:
        """
        基准测试预处理性能
        
        Args:
            image_paths: 图像路径列表
            preprocessor: 预处理器
            batch_sizes: 批大小列表
            num_workers_list: 工作进程数列表
            
        Returns:
            基准测试结果
        """
        if batch_sizes is None:
            batch_sizes = [16, 32, 64, 128]
        
        if num_workers_list is None:
            num_workers_list = [0, 2, 4, 8]
        
        results = {}
        sample_paths = image_paths[:min(1000, len(image_paths))]  # 限制样本数量
        labels = [0] * len(sample_paths)  # 虚拟标签
        
        for batch_size in batch_sizes:
            for num_workers in num_workers_list:
                config_name = f"batch_{batch_size}_workers_{num_workers}"
                
                try:
                    # 创建数据加载器
                    dataloader = self.create_optimized_dataloader(
                        sample_paths, labels, preprocessor,
                        batch_size=batch_size,
                        num_workers=num_workers
                    )
                    
                    # 测试性能
                    metrics = self._measure_dataloader_performance(dataloader)
                    results[config_name] = metrics
                    
                    logger.info(f"配置 {config_name}: {metrics.images_per_second:.2f} images/sec")
                    
                except Exception as e:
                    logger.error(f"基准测试失败 {config_name}: {e}")
                    results[config_name] = None
        
        # 找到最优配置
        best_config = max(
            [(k, v) for k, v in results.items() if v is not None],
            key=lambda x: x[1].images_per_second,
            default=(None, None)
        )
        
        return {
            'results': results,
            'best_config': best_config[0] if best_config[0] else None,
            'best_performance': best_config[1] if best_config[1] else None
        }
    
    def _measure_dataloader_performance(self, dataloader: DataLoader) -> PerformanceMetrics:
        """测量数据加载器性能"""
        start_time = time.time()
        start_memory = self.memory_optimizer._get_memory_usage()
        
        total_images = 0
        batch_count = 0
        max_batches = 10  # 限制测试批次数
        
        try:
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                
                total_images += len(images)
                batch_count += 1
                
                # 模拟一些处理
                _ = images.mean()
        
        except Exception as e:
            logger.warning(f"性能测试中断: {e}")
        
        end_time = time.time()
        end_memory = self.memory_optimizer._get_memory_usage()
        
        processing_time = end_time - start_time
        images_per_second = total_images / processing_time if processing_time > 0 else 0
        
        return PerformanceMetrics(
            total_images=total_images,
            processing_time=processing_time,
            memory_usage_mb=end_memory - start_memory,
            images_per_second=images_per_second,
            avg_image_size_mb=0,  # 简化
            cache_hit_rate=0      # 简化
        )
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """根据硬件配置优化参数"""
        # 获取系统信息
        cpu_count_val = cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        has_gpu = torch.cuda.is_available()
        
        # 推荐配置
        recommendations = {
            'cpu_cores': cpu_count_val,
            'memory_gb': memory_gb,
            'has_gpu': has_gpu,
            'recommended_batch_size': 32 if memory_gb >= 8 else 16,
            'recommended_num_workers': min(cpu_count_val, 8),
            'recommended_cache_size_mb': min(int(memory_gb * 1024 * 0.1), 1024),
            'pin_memory': has_gpu,
            'prefetch_factor': 2 if memory_gb >= 16 else 1
        }
        
        if has_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            recommendations['gpu_memory_gb'] = gpu_memory
            recommendations['recommended_batch_size'] = min(
                recommendations['recommended_batch_size'],
                int(gpu_memory * 32)  # 经验公式
            )
        
        return recommendations

# 便捷函数
def create_preprocessing_optimizer() -> PreprocessingOptimizer:
    """创建预处理优化器"""
    return PreprocessingOptimizer()

def create_optimized_dataloader(image_paths: List[str],
                              labels: List[int],
                              preprocessor: PlantDiseasePreprocessor,
                              **kwargs) -> DataLoader:
    """便捷函数：创建优化数据加载器"""
    optimizer = create_preprocessing_optimizer()
    return optimizer.create_optimized_dataloader(image_paths, labels, preprocessor, **kwargs)

if __name__ == "__main__":
    # 测试预处理优化器
    print("🧪 预处理性能优化器测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试优化器创建
        print("📋 测试优化器创建...")
        optimizer = create_preprocessing_optimizer()
        print(f"✅ 优化器创建成功")
        
        # 测试硬件优化建议
        print(f"\n🔧 测试硬件优化建议...")
        hardware_config = optimizer.optimize_for_hardware()
        print(f"✅ 硬件配置分析完成:")
        print(f"   CPU核心数: {hardware_config['cpu_cores']}")
        print(f"   内存: {hardware_config['memory_gb']:.1f} GB")
        print(f"   GPU可用: {hardware_config['has_gpu']}")
        print(f"   推荐批大小: {hardware_config['recommended_batch_size']}")
        print(f"   推荐工作进程: {hardware_config['recommended_num_workers']}")
        
        # 测试内存优化
        print(f"\n💾 测试内存优化...")
        memory_optimizer = MemoryOptimizer()
        memory_stats = memory_optimizer.optimize_memory()
        print(f"✅ 内存优化完成:")
        print(f"   优化前: {memory_stats['before_mb']:.1f} MB")
        print(f"   优化后: {memory_stats['after_mb']:.1f} MB")
        print(f"   释放内存: {memory_stats['freed_mb']:.1f} MB")
        
        # 测试图像缓存
        print(f"\n🗄️ 测试图像缓存...")
        cache = ImageCache(max_size_mb=100)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 测试缓存操作
        cache.put("test_image", test_image)
        cached_image = cache.get("test_image")
        
        print(f"✅ 图像缓存测试完成:")
        print(f"   缓存大小: {cache.current_size / 1024 / 1024:.2f} MB")
        print(f"   命中率: {cache.get_hit_rate():.2%}")
        print(f"   图像匹配: {np.array_equal(test_image, cached_image)}")
        
        # 测试批量处理器
        print(f"\n⚡ 测试批量处理器...")
        from src.image_preprocessing import create_plant_preprocessor, PreprocessingMode
        
        batch_processor = BatchProcessor(batch_size=4, num_workers=2)
        preprocessor = create_plant_preprocessor(PreprocessingMode.VALIDATION)
        
        # 创建虚拟图像路径（实际测试中应使用真实路径）
        dummy_paths = ["dummy_path"] * 4
        print(f"✅ 批量处理器创建完成:")
        print(f"   批大小: {batch_processor.batch_size}")
        print(f"   工作进程: {batch_processor.num_workers}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ 预处理性能优化器测试完成")