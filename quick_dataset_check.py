#!/usr/bin/env python3
"""
快速数据集检查工具
只检查基本结构和统计信息，不深入分析每个文件
"""

import os
from pathlib import Path
from collections import defaultdict

def quick_check_directory(path: str, max_depth: int = 2) -> dict:
    """快速检查目录结构"""
    if not os.path.exists(path):
        return {'status': 'not_found', 'message': f'路径不存在: {path}'}
    
    result = {
        'status': 'found',
        'path': path,
