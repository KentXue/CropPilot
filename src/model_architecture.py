#!/usr/bin/env python3
"""
EfficientNet-B4模型架构
实现植物病害识别的深度学习模型架构
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.hub import load_state_dict_from_url
    import torchvision.models as models
    from efficientnet_pytorch import EfficientNet
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  缺少依赖: {e}")
    DEPENDENCIES_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseEfficientNet(nn.Module):
    """植物病害识别EfficientNet模型"""
    
    def __init__(self, 
                 num_classes: int = 38,
                 model_name: str = 'efficientnet-b4',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 drop_connect_rate: float = 0.2):
        """
        初始化EfficientNet模型
        
        Args:
            num_classes: 分类类别数
            model_name: 模型名称
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout率
            drop_connect_rate: DropConnect率
        """
        super(PlantDiseaseEfficientNet, self).__init__()
        
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("需要安装必要依赖")
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # 加载EfficientNet骨干网络
        try:
            if pretrained:
                self.backbone = EfficientNet.from_pretrained(
                    model_name, 
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                    drop_connect_rate=drop_connect_rate
                )
                logger.info(f"成功加载预训练权重: {model_name}")
            else:
                self.backbone = EfficientNet.from_name(
                    model_name,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                    drop_connect_rate=drop_connect_rate
                )
        except Exception as e:
            logger.warning(f"加载预训练权重失败: {e}")
            logger.info("使用随机初始化权重")
            self.backbone = EfficientNet.from_name(
                model_name,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                drop_connect_rate=drop_connect_rate
            )
        
        # 获取特征维度
        self.feature_dim = self.backbone._fc.in_features
        
        # 替换分类头
        self.backbone._fc = nn.Identity()
        
        # 创建自定义分类头
        self.classifier = self._create_classifier_head()
        
        # 初始化权重
        self._initialize_weights()
        
        logger.info(f"PlantDiseaseEfficientNet初始化完成: {model_name}, 类别数: {num_classes}")
    
    def _create_classifier_head(self) -> nn.Module:
        """创建分类头"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate / 4),
            nn.Linear(256, self.num_classes)
        )
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 提取特征
        features = self.backbone.extract_features(x)
        
        # 分类
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量"""
        features = self.backbone.extract_features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        features = features.flatten(1)
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'dropout_rate': self.dropout_rate
        }

class MultiScaleEfficientNet(nn.Module):
    """多尺度EfficientNet模型"""
    
    def __init__(self, 
                 num_classes: int = 38,
                 scales: List[str] = None,
                 fusion_method: str = 'attention'):
        """
        初始化多尺度模型
        
        Args:
            num_classes: 分类类别数
            scales: 尺度列表
            fusion_method: 融合方法 ('concat', 'attention', 'weighted')
        """
        super(MultiScaleEfficientNet, self).__init__()
        
        if scales is None:
            scales = ['efficientnet-b2', 'efficientnet-b4', 'efficientnet-b5']
        
        self.scales = scales
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # 创建多个尺度的模型
        self.scale_models = nn.ModuleDict()
        self.feature_dims = []
        
        for scale in scales:
            model = PlantDiseaseEfficientNet(
                num_classes=num_classes,
                model_name=scale,
                pretrained=True
            )
            # 移除分类头，只保留特征提取
            model.classifier = nn.Identity()
            self.scale_models[scale] = model
            self.feature_dims.append(model.feature_dim)
        
        # 创建融合层
        self.fusion_layer = self._create_fusion_layer()
        
        # 最终分类器
        self.final_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self._get_fusion_dim(), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"MultiScaleEfficientNet初始化完成: {len(scales)} 个尺度")
    
    def _create_fusion_layer(self) -> nn.Module:
        """创建特征融合层"""
        if self.fusion_method == 'concat':
            return nn.Identity()
        elif self.fusion_method == 'attention':
            total_dim = sum(self.feature_dims)
            return nn.Sequential(
                nn.Linear(total_dim, total_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(total_dim // 4, len(self.scales)),
                nn.Softmax(dim=1)
            )
        elif self.fusion_method == 'weighted':
            return nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
    
    def _get_fusion_dim(self) -> int:
        """获取融合后的特征维度"""
        if self.fusion_method == 'concat':
            return sum(self.feature_dims)
        else:
            return max(self.feature_dims)  # 假设使用最大维度
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 提取多尺度特征
        scale_features = []
        for scale, model in self.scale_models.items():
            features = model.extract_features(x)
            scale_features.append(features)
        
        # 特征融合
        if self.fusion_method == 'concat':
            fused_features = torch.cat(scale_features, dim=1)
        elif self.fusion_method == 'attention':
            stacked_features = torch.stack(scale_features, dim=1)  # [B, N, D]
            attention_weights = self.fusion_layer(torch.cat(scale_features, dim=1))
            attention_weights = attention_weights.unsqueeze(-1)  # [B, N, 1]
            fused_features = (stacked_features * attention_weights).sum(dim=1)
        elif self.fusion_method == 'weighted':
            stacked_features = torch.stack(scale_features, dim=1)  # [B, N, D]
            weights = F.softmax(self.fusion_layer, dim=0).unsqueeze(0).unsqueeze(-1)
            fused_features = (stacked_features * weights).sum(dim=1)
        
        # 最终分类
        output = self.final_classifier(fused_features)
        
        return output

class EnsembleEfficientNet(nn.Module):
    """集成EfficientNet模型"""
    
    def __init__(self, 
                 num_classes: int = 38,
                 model_configs: List[Dict[str, Any]] = None,
                 ensemble_method: str = 'voting'):
        """
        初始化集成模型
        
        Args:
            num_classes: 分类类别数
            model_configs: 模型配置列表
            ensemble_method: 集成方法 ('voting', 'weighted', 'stacking')
        """
        super(EnsembleEfficientNet, self).__init__()
        
        if model_configs is None:
            model_configs = [
                {'model_name': 'efficientnet-b3', 'dropout_rate': 0.3},
                {'model_name': 'efficientnet-b4', 'dropout_rate': 0.3},
                {'model_name': 'efficientnet-b5', 'dropout_rate': 0.2}
            ]
        
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        self.num_models = len(model_configs)
        
        # 创建多个模型
        self.models = nn.ModuleList()
        for i, config in enumerate(model_configs):
            model = PlantDiseaseEfficientNet(
                num_classes=num_classes,
                **config
            )
            self.models.append(model)
        
        # 创建集成层
        if ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        elif ensemble_method == 'stacking':
            # 堆叠学习器
            self.stacking_classifier = nn.Sequential(
                nn.Linear(num_classes * self.num_models, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        logger.info(f"EnsembleEfficientNet初始化完成: {self.num_models} 个模型")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取所有模型的输出
        model_outputs = []
        for model in self.models:
            output = model(x)
            model_outputs.append(output)
        
        # 集成预测
        if self.ensemble_method == 'voting':
            # 简单平均
            ensemble_output = torch.stack(model_outputs, dim=0).mean(dim=0)
        elif self.ensemble_method == 'weighted':
            # 加权平均
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_outputs = []
            for i, output in enumerate(model_outputs):
                weighted_outputs.append(output * weights[i])
            ensemble_output = torch.stack(weighted_outputs, dim=0).sum(dim=0)
        elif self.ensemble_method == 'stacking':
            # 堆叠学习
            stacked_input = torch.cat(model_outputs, dim=1)
            ensemble_output = self.stacking_classifier(stacked_input)
        
        return ensemble_output

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_type: str, 
                    num_classes: int = 38,
                    **kwargs) -> nn.Module:
        """
        创建模型
        
        Args:
            model_type: 模型类型
            num_classes: 分类类别数
            **kwargs: 其他参数
            
        Returns:
            创建的模型
        """
        if model_type == 'efficientnet':
            return PlantDiseaseEfficientNet(num_classes=num_classes, **kwargs)
        elif model_type == 'multiscale':
            return MultiScaleEfficientNet(num_classes=num_classes, **kwargs)
        elif model_type == 'ensemble':
            return EnsembleEfficientNet(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        
        info = {
            'model_type': type(model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'num_classes': getattr(model, 'num_classes', 'unknown')
        }
        
        # 添加模型特定信息
        if hasattr(model, 'get_model_info'):
            info.update(model.get_model_info())
        
        return info

class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """统计模型参数"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    @staticmethod
    def freeze_backbone(model: PlantDiseaseEfficientNet, freeze: bool = True):
        """冻结/解冻骨干网络"""
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"骨干网络已{'冻结' if freeze else '解冻'}")
    
    @staticmethod
    def get_layer_names(model: nn.Module) -> List[str]:
        """获取模型层名称"""
        layer_names = []
        for name, _ in model.named_modules():
            if name:  # 排除根模块
                layer_names.append(name)
        return layer_names
    
    @staticmethod
    def calculate_model_flops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> int:
        """计算模型FLOPs（简化实现）"""
        # 这是一个简化的实现，实际应该使用专门的工具如thop
        total_params = sum(p.numel() for p in model.parameters())
        # 粗略估计：每个参数大约对应2个FLOPs
        estimated_flops = total_params * 2 * input_size[0]  # 乘以batch size
        return estimated_flops

# 便捷函数
def create_plant_disease_model(model_type: str = 'efficientnet',
                             num_classes: int = 38,
                             **kwargs) -> nn.Module:
    """创建植物病害识别模型"""
    return ModelFactory.create_model(model_type, num_classes, **kwargs)

def load_pretrained_model(model_path: str, 
                         model_type: str = 'efficientnet',
                         num_classes: int = 38,
                         **kwargs) -> nn.Module:
    """加载预训练模型"""
    model = create_plant_disease_model(model_type, num_classes, **kwargs)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"模型权重已从 {model_path} 加载")
    else:
        logger.warning(f"模型文件不存在: {model_path}")
    
    return model

if __name__ == "__main__":
    # 测试模型架构
    print("🧪 EfficientNet模型架构测试")
    print("=" * 60)
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 缺少必要依赖，无法运行测试")
        sys.exit(1)
    
    try:
        # 测试基础EfficientNet模型
        print("📋 测试基础EfficientNet模型...")
        model = create_plant_disease_model(
            'efficientnet', 
            num_classes=38,
            pretrained=True  # 使用预训练权重
        )
        
        model_info = ModelFactory.get_model_info(model)
        print(f"✅ EfficientNet模型创建成功:")
        print(f"   模型类型: {model_info['model_type']}")
        print(f"   总参数数: {model_info['total_parameters']:,}")
        print(f"   可训练参数: {model_info['trainable_parameters']:,}")
        print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
        print(f"   类别数: {model_info['num_classes']}")
        
        # 测试前向传播
        print(f"\n🔍 测试前向传播...")
        test_input = torch.randn(2, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            features = model.extract_features(test_input)
        
        print(f"✅ 前向传播测试完成:")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   特征形状: {features.shape}")
        print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        # 测试多尺度模型
        print(f"\n📏 测试多尺度模型...")
        multiscale_model = create_plant_disease_model(
            'multiscale', 
            num_classes=38,
            scales=['efficientnet-b2', 'efficientnet-b4']
        )
        
        multiscale_info = ModelFactory.get_model_info(multiscale_model)
        print(f"✅ 多尺度模型创建成功:")
        print(f"   总参数数: {multiscale_info['total_parameters']:,}")
        print(f"   模型大小: {multiscale_info['model_size_mb']:.2f} MB")
        
        # 测试集成模型
        print(f"\n🎯 测试集成模型...")
        ensemble_model = create_plant_disease_model(
            'ensemble',
            num_classes=38,
            model_configs=[
                {'model_name': 'efficientnet-b3', 'dropout_rate': 0.3},
                {'model_name': 'efficientnet-b4', 'dropout_rate': 0.2}
            ]
        )
        
        ensemble_info = ModelFactory.get_model_info(ensemble_model)
        print(f"✅ 集成模型创建成功:")
        print(f"   总参数数: {ensemble_info['total_parameters']:,}")
        print(f"   模型大小: {ensemble_info['model_size_mb']:.2f} MB")
        
        # 测试模型工具
        print(f"\n🔧 测试模型工具...")
        param_stats = ModelUtils.count_parameters(model)
        layer_names = ModelUtils.get_layer_names(model)
        
        print(f"✅ 模型工具测试完成:")
        print(f"   参数统计: {param_stats}")
        print(f"   层数量: {len(layer_names)}")
        print(f"   前5层: {layer_names[:5]}")
        
        # 测试冻结/解冻
        print(f"\n❄️ 测试冻结/解冻功能...")
        ModelUtils.freeze_backbone(model, freeze=True)
        frozen_params = ModelUtils.count_parameters(model)
        
        ModelUtils.freeze_backbone(model, freeze=False)
        unfrozen_params = ModelUtils.count_parameters(model)
        
        print(f"✅ 冻结/解冻测试完成:")
        print(f"   冻结后可训练参数: {frozen_params['trainable']:,}")
        print(f"   解冻后可训练参数: {unfrozen_params['trainable']:,}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ EfficientNet模型架构测试完成")