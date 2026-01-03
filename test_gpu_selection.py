#!/usr/bin/env python3
"""
GPU选择测试脚本
"""

import torch
import time

def test_gpu_selection():
    """测试GPU选择逻辑"""
    print("🔍 GPU选择测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU:")
    
    # 显示所有GPU信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        
        print(f"  GPU {i}: {gpu_name}")
        print(f"    显存: {memory_gb:.1f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    多处理器: {props.multi_processor_count}")
        print()
    
    # 选择最佳GPU
    best_gpu = 0
    if gpu_count > 1:
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            # RTX、GTX等独立显卡优先
            if any(keyword in gpu_name.upper() for keyword in ['RTX', 'GTX', 'TESLA', 'QUADRO']):
                best_gpu = i
                break
    
    print(f"🎯 选择的GPU: GPU {best_gpu} - {torch.cuda.get_device_name(best_gpu)}")
    
    # 测试选定GPU的性能
    device = torch.device(f'cuda:{best_gpu}')
    
    print(f"\n🧪 测试GPU {best_gpu}性能...")
    
    # 矩阵乘法测试
    sizes = [1000, 2000, 3000]
    for size in sizes:
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        start_time = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"  {size}x{size} 矩阵乘法: {(end_time-start_time)*1000:.1f}ms")
        
        del x, y, z
    
    # 显存使用测试
    print(f"\n💾 显存使用测试...")
    allocated = torch.cuda.memory_allocated(best_gpu) / 1024**3
    cached = torch.cuda.memory_reserved(best_gpu) / 1024**3
    total = torch.cuda.get_device_properties(best_gpu).total_memory / 1024**3
    
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已缓存: {cached:.2f} GB")
    print(f"  总显存: {total:.2f} GB")
    print(f"  使用率: {(allocated/total)*100:.1f}%")
    
    torch.cuda.empty_cache()
    print(f"\n✅ GPU {best_gpu} 工作正常")

if __name__ == "__main__":
    test_gpu_selection()