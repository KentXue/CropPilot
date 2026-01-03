#!/usr/bin/env python3
"""
GPU压力测试 - 让GPU使用率明显可见
"""

import torch
import time
from src.model_architecture import create_plant_disease_model

def gpu_stress_test():
    """GPU压力测试"""
    if not torch.cuda.is_available():
        print("❌ GPU不可用")
        return
    
    print("🔥 GPU压力测试开始")
    print("现在应该能在任务管理器中看到GPU使用率上升")
    print("=" * 50)
    
    device = torch.device('cuda')
    
    # 创建大模型
    model = create_plant_disease_model('efficientnet', pretrained=False).to(device)
    model.train()
    
    # 创建大批量数据
    batch_size = 32  # 增大批大小
    data = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 38, (batch_size,), device=device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"批大小: {batch_size}")
    print(f"数据在GPU上: {data.device}")
    
    # 显存使用
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"当前显存使用: {allocated:.2f} GB")
    
    print("\n开始高强度训练循环...")
    print("请查看任务管理器的GPU使用率")
    
    # 高强度训练循环
    for i in range(100):
        start_time = time.time()
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 同步GPU
        torch.cuda.synchronize()
        
        end_time = time.time()
        
        if i % 10 == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"步骤 {i:3d}: 损失={loss.item():.4f}, 时间={end_time-start_time:.3f}s, 显存={allocated:.2f}GB")
    
    print("\n✅ GPU压力测试完成")
    print("如果任务管理器显示GPU使用率很高，说明GPU正常工作")

if __name__ == "__main__":
    gpu_stress_test()