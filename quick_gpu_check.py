#!/usr/bin/env python3
"""
快速GPU使用检查
"""

import torch
import time

def check_gpu_usage():
    """检查GPU使用情况"""
    if not torch.cuda.is_available():
        print("❌ GPU不可用")
        return
    
    print("🔍 GPU使用情况检查")
    print("=" * 40)
    
    # GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"GPU: {gpu_name}")
    print(f"总显存: {total_memory:.1f} GB")
    
    # 当前使用情况
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"已分配显存: {allocated:.2f} GB")
    print(f"已缓存显存: {cached:.2f} GB")
    print(f"使用率: {(allocated/total_memory)*100:.1f}%")
    
    # 测试GPU计算
    print("\n🧪 GPU计算测试...")
    device = torch.device('cuda')
    
    # 创建大张量测试
    start_time = time.time()
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.mm(x, y)  # 矩阵乘法
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"矩阵乘法耗时: {(end_time-start_time)*1000:.1f}ms")
    
    # 检查使用后的显存
    allocated_after = torch.cuda.memory_allocated(0) / 1024**3
    print(f"计算后显存: {allocated_after:.2f} GB")
    
    # 清理
    del x, y, z
    torch.cuda.empty_cache()
    
    print("✅ GPU工作正常")

if __name__ == "__main__":
    check_gpu_usage()