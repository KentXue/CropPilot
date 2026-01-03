#!/usr/bin/env python3
"""
GPU使用测试脚本
验证RTX 4050是否被正确使用进行训练
"""

import torch
import torch.nn as nn
import time
import psutil
import os

def test_gpu_selection():
    """测试GPU选择逻辑"""
    print("🔍 GPU选择测试")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU:")
    
    best_gpu = 0
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")
        
        # RTX、GTX等独立显卡优先
        if any(keyword in gpu_name.upper() for keyword in ['RTX', 'GTX', 'TESLA', 'QUADRO']):
            best_gpu = i
            print(f"    ✅ 选择此GPU用于训练")
        else:
            print(f"    ⚪ 集成显卡，不优先选择")
    
    selected_device = torch.device(f'cuda:{best_gpu}')
    print(f"\n🎯 最终选择: {selected_device} - {torch.cuda.get_device_name(best_gpu)}")
    
    return selected_device

def test_gpu_computation(device, duration=10):
    """测试GPU计算负载"""
    print(f"\n🚀 GPU计算测试 (设备: {device})")
    print("=" * 50)
    
    # 创建一个简单的神经网络
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"开始 {duration} 秒的GPU计算测试...")
    
    start_time = time.time()
    step = 0
    
    while time.time() - start_time < duration:
        # 生成随机数据
        batch_size = 64
        x = torch.randn(batch_size, 1000, device=device)
        y = torch.randn(batch_size, 100, device=device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        step += 1
        
        # 每100步显示一次状态
        if step % 100 == 0:
            elapsed = time.time() - start_time
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
            
            print(f"步骤 {step:4d}: 损失={loss.item():.4f}, "
                  f"时间={elapsed:.1f}s, 显存={gpu_memory:.2f}GB")
    
    total_time = time.time() - start_time
    print(f"\n✅ GPU计算测试完成:")
    print(f"   总步骤: {step}")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   平均步骤时间: {total_time/step*1000:.2f}ms")
    print(f"   最终显存使用: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB")

def monitor_system_resources():
    """监控系统资源使用"""
    print(f"\n📊 系统资源监控")
    print("=" * 50)
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU使用率: {cpu_percent:.1f}%")
    
    # 内存使用
    memory = psutil.virtual_memory()
    print(f"内存使用: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    
    # GPU信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.memory_allocated(i) / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"GPU {i} ({gpu_name}):")
            print(f"  显存使用: {gpu_memory:.2f}GB / {gpu_memory_total:.1f}GB")

def main():
    """主函数"""
    print("🧪 GPU使用验证测试")
    print("=" * 80)
    
    # 1. 测试GPU选择
    device = test_gpu_selection()
    
    # 2. 监控系统资源
    monitor_system_resources()
    
    # 3. 测试GPU计算
    if torch.cuda.is_available():
        test_gpu_computation(device, duration=15)
        
        # 4. 再次监控资源
        print(f"\n📊 测试后系统资源:")
        monitor_system_resources()
    else:
        print("❌ 无法进行GPU计算测试")
    
    print("\n" + "=" * 80)
    print("✅ GPU使用验证测试完成")

if __name__ == "__main__":
    main()