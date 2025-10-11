"""
优化器模块 - 实际训练版本
================================

本模块提供优化器的工厂函数，使用 PyTorch 内置优化器。
代码精简，性能优化，适合实际训练使用。

如果想了解优化器的实现原理，请查看 tutorial/optimizer_from_scratch.py

支持的优化器：
- AdamW: 带解耦权重衰减的 Adam
- Adam: 标准 Adam 优化器
- SGD: 随机梯度下降（可选动量）

作者：AnotherLLMFromScratch 项目
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam, SGD


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    **kwargs
) -> Optimizer:
    """
    创建优化器的工厂函数
    
    参数：
        model: 要优化的模型
        optimizer_type: 优化器类型 ('adamw', 'adam', 'sgd')
        lr: 学习率
        weight_decay: 权重衰减系数（L2 正则化）
        betas: Adam 系列优化器的 beta 参数
        eps: 数值稳定性常数
        momentum: SGD 的动量系数
        **kwargs: 其他优化器参数
    
    返回：
        配置好的优化器实例
    
    使用示例：
        >>> optimizer = create_optimizer(model, 'adamw', lr=1e-3)
        >>> optimizer = create_optimizer(model, 'sgd', lr=0.01, momentum=0.9)
    """
    optimizer_type = optimizer_type.lower()
    
    # 获取模型参数（过滤不需要梯度的参数）
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == 'adamw':
        # AdamW: Adam with decoupled weight decay
        # 最常用于 Transformer 模型
        optimizer = AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_type == 'adam':
        # 标准 Adam 优化器
        optimizer = Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_type == 'sgd':
        # SGD with momentum
        optimizer = SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_bias: bool = True,
    no_decay_norm: bool = True
) -> List[Dict[str, Any]]:
    """
    为不同类型的参数创建参数组，应用不同的权重衰减策略
    
    最佳实践：
    - 通常对 bias 和 LayerNorm 参数不应用权重衰减
    - 对权重矩阵应用权重衰减
    
    参数：
        model: 模型
        weight_decay: 权重衰减系数
        no_decay_bias: bias 参数是否不应用权重衰减
        no_decay_norm: normalization 层参数是否不应用权重衰减
    
    返回：
        参数组列表，每组有不同的配置
    
    使用示例：
        >>> param_groups = get_parameter_groups(model, weight_decay=0.01)
        >>> optimizer = AdamW(param_groups, lr=1e-3)
    """
    # 需要权重衰减的参数
    decay_params = []
    # 不需要权重衰减的参数
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 判断是否需要权重衰减
        apply_decay = True
        
        # bias 参数通常不需要权重衰减
        if no_decay_bias and 'bias' in name:
            apply_decay = False
        
        # normalization 层参数通常不需要权重衰减
        if no_decay_norm and any(x in name.lower() for x in ['norm', 'ln', 'bn']):
            apply_decay = False
        
        if apply_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    # 创建参数组
    param_groups = [
        {
            'params': decay_params,
            'weight_decay': weight_decay,
            'name': 'decay'
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'name': 'no_decay'
        }
    ]
    
    return param_groups


def configure_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    use_parameter_groups: bool = True,
    **kwargs
) -> Optimizer:
    """
    配置优化器的便捷函数（带参数分组）
    
    这是 create_optimizer 的增强版本，自动处理参数分组。
    
    参数：
        model: 要优化的模型
        optimizer_type: 优化器类型
        lr: 学习率
        weight_decay: 权重衰减系数
        use_parameter_groups: 是否使用参数分组（自动处理 bias 和 norm）
        **kwargs: 其他优化器参数
    
    返回：
        配置好的优化器实例
    
    使用示例：
        >>> # 自动参数分组（推荐）
        >>> optimizer = configure_optimizer(model, 'adamw', lr=1e-3)
        >>> 
        >>> # 不使用参数分组
        >>> optimizer = configure_optimizer(model, 'adamw', lr=1e-3, 
        ...                                 use_parameter_groups=False)
    """
    if use_parameter_groups:
        # 使用参数分组
        param_groups = get_parameter_groups(model, weight_decay=weight_decay)
        
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adamw':
            optimizer = AdamW(param_groups, lr=lr, **kwargs)
        elif optimizer_type == 'adam':
            optimizer = Adam(param_groups, lr=lr, **kwargs)
        elif optimizer_type == 'sgd':
            optimizer = SGD(param_groups, lr=lr, **kwargs)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    else:
        # 不使用参数分组，直接创建
        optimizer = create_optimizer(
            model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    return optimizer


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """演示优化器的使用"""
    print("=" * 70)
    print("优化器模块使用示例")
    print("=" * 70)
    
    # 创建一个简单的模型
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.LayerNorm(50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    print("\n方法 1: 使用 create_optimizer（简单）")
    print("-" * 70)
    optimizer1 = create_optimizer(model, 'adamw', lr=1e-3, weight_decay=0.01)
    print(f"创建的优化器: {type(optimizer1).__name__}")
    print(f"参数组数量: {len(optimizer1.param_groups)}")
    
    print("\n方法 2: 使用 configure_optimizer（推荐，带参数分组）")
    print("-" * 70)
    optimizer2 = configure_optimizer(
        model,
        optimizer_type='adamw',
        lr=1e-3,
        weight_decay=0.01,
        use_parameter_groups=True
    )
    print(f"创建的优化器: {type(optimizer2).__name__}")
    print(f"参数组数量: {len(optimizer2.param_groups)}")
    
    for i, group in enumerate(optimizer2.param_groups):
        print(f"  组 {i} ({group['name']}): "
              f"{len(group['params'])} 个参数, "
              f"weight_decay={group['weight_decay']}")
    
    print("\n方法 3: 手动创建参数组")
    print("-" * 70)
    param_groups = get_parameter_groups(model, weight_decay=0.01)
    optimizer3 = AdamW(param_groups, lr=1e-3)
    print(f"创建的优化器: {type(optimizer3).__name__}")
    
    print("\n" + "=" * 70)
    print("提示：查看 tutorial/optimizer_from_scratch.py 了解优化器的实现原理")
    print("=" * 70)
