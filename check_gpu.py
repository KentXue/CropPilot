#!/usr/bin/env python3
"""
GPU检测和优化脚本
检查CUDA可用性并优化GPU训练设置
"""

import os
import sys

try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ PyTorch未安装")
    TORCH_AVAILABLE = False

def check_gpu_availability():
    """检查GPU可用性"""
    print("🔍 GPU可用性检查")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch未安装，无法检查GPU")
        return False
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {'✅ 是' if cuda_available else '❌ 否'}")
    
    if not cuda_available:
        print("\n可能的原因:")
        print("1. 没有安装CUDA兼容的PyTorch版本")
        print("2. CUDA驱动程序未正确安装")
        print("3. GPU不支持CUDA")
        print("\n建议:")
        print("访问 https://pytorch.org/ 安装CUDA版本的PyTorch")
        return False
    
    # 获取GPU信息
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 测试GPU内存
    if gpu_count > 0:
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(1000, 1000, device=device)
            print(f"✅ GPU内存测试通过")
            
            # 显示当前GPU内存使用
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU内存使用: {allocated:.2f} GB (已分配) / {cached:.2f} GB (已缓存)")
            
            del test_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU内存测试失败: {e}")
            return False
    
    return True

def optimize_gpu_settings():
    """优化GPU训练设置"""
    print("\n⚙️ GPU训练优化建议")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ GPU不可用，无法提供优化建议")
        return {}
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_props.total_memory / 1024**3
    
    print(f"GPU: {gpu_props.name}")
    print(f"显存: {gpu_memory_gb:.1f} GB")
    print(f"计算能力: {gpu_props.major}.{gpu_props.minor}")
    
    # 根据显存大小推荐批大小
    if gpu_memory_gb >= 24:
        recommended_batch_size = 64
        mixed_precision = True
        print("🚀 高端GPU配置")
    elif gpu_memory_gb >= 12:
        recommended_batch_size = 32
        mixed_precision = True
        print("💪 中高端GPU配置")
    elif gpu_memory_gb >= 8:
        recommended_batch_size = 16
        mixed_precision = True
        print("👍 中端GPU配置")
    elif gpu_memory_gb >= 4:
        recommended_batch_size = 8
        mixed_precision = True
        print("⚠️ 入门级GPU配置")
    else:
        recommended_batch_size = 4
        mixed_precision = False
        print("🔥 低显存GPU配置")
    
    recommendations = {
        'batch_size': recommended_batch_size,
        'mixed_precision': mixed_precision,
        'num_workers': min(8, os.cpu_count()),
        'pin_memory': True,
        'device': 'cuda'
    }
    
    print(f"\n推荐设置:")
    print(f"  批大小: {recommended_batch_size}")
    print(f"  混合精度: {'启用' if mixed_precision else '禁用'}")
    print(f"  数据加载进程: {recommendations['num_workers']}")
    print(f"  内存固定: {'启用' if recommendations['pin_memory'] else '禁用'}")
    
    # 额外优化建议
    print(f"\n额外优化建议:")
    if gpu_memory_gb < 8:
        print("  - 考虑使用梯度累积来模拟更大的批大小")
        print("  - 使用较小的模型如efficientnet-b2或b3")
    
    if mixed_precision:
        print("  - 启用混合精度训练可以节省约50%显存")
        print("  - 可能需要调整学习率（通常增加1.5-2倍）")
    
    print("  - 使用torch.backends.cudnn.benchmark = True加速训练")
    print("  - 定期清理GPU缓存: torch.cuda.empty_cache()")
    
    return recommendations

def create_gpu_optimized_config():
    """创建GPU优化的训练配置"""
    if not torch.cuda.is_available():
        print("\n❌ GPU不可用，无法创建GPU优化配置")
        return
    
    recommendations = optimize_gpu_settings()
    
    config = {
        "num_epochs": 50,
        "batch_size": recommendations['batch_size'],
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "model_type": "efficientnet",
        "model_name": "efficientnet-b4",
        "num_classes": 38,
        "pretrained": True,
        "optimizer_type": "adamw",
        "scheduler_type": "cosine",
        "early_stopping": True,
        "patience": 10,
        "min_delta": 0.001,
        "save_dir": "checkpoints/plant_disease_gpu",
        "device": "cuda",
        "mixed_precision": recommendations['mixed_precision'],
        "gradient_clip_norm": 1.0,
        "label_smoothing": 0.1
    }
    
    # 保存配置文件
    import json
    config_path = "gpu_training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ GPU优化配置已保存: {config_path}")
    print("使用方法: python train_model.py --config gpu_training_config.json")

def benchmark_gpu_performance():
    """GPU性能基准测试"""
    if not torch.cuda.is_available():
        print("\n❌ GPU不可用，无法进行性能测试")
        return
    
    print("\n🏃 GPU性能基准测试")
    print("=" * 50)
    
    device = torch.device('cuda')
    
    # 测试不同批大小的性能
    batch_sizes = [8, 16, 32, 64]
    input_size = (3, 224, 224)
    
    print("测试EfficientNet-B4前向传播性能:")
    
    try:
        from src.model_architecture import create_plant_disease_model
        model = create_plant_disease_model('efficientnet', pretrained=False).to(device)
        model.eval()
        
        import time
        
        for batch_size in batch_sizes:
            try:
                # 预热
                with torch.no_grad():
                    dummy_input = torch.randn(batch_size, *input_size, device=device)
                    for _ in range(5):
                        _ = model(dummy_input)
                
                # 性能测试
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(20):
                        _ = model(dummy_input)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 20
                throughput = batch_size / avg_time
                
                print(f"  批大小 {batch_size:2d}: {avg_time*1000:.1f}ms/batch, {throughput:.1f} images/sec")
                
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  批大小 {batch_size:2d}: ❌ 显存不足")
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        del model
        torch.cuda.empty_cache()
        
    except ImportError:
        print("❌ 无法导入模型，跳过性能测试")

def main():
    """主函数"""
    print("🔧 CropPilot GPU检测和优化工具")
    print("=" * 60)
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        # 优化设置建议
        optimize_gpu_settings()
        
        # 创建GPU优化配置
        create_gpu_optimized_config()
        
        # 性能基准测试
        benchmark_gpu_performance()
        
        print("\n🎉 GPU检测和优化完成!")
        print("现在可以使用GPU进行高效训练了")
    else:
        print("\n💡 建议:")
        print("1. 安装CUDA版本的PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("2. 确保NVIDIA驱动程序已正确安装")
        print("3. 重启系统后重新测试")

if __name__ == "__main__":
    main()