"""
Trainer 模块
==============

提供多种训练范式的训练器实现。

训练器架构：
- BaseTrainer: 抽象基类，包含所有通用训练逻辑
- SFTTrainer: 监督微调训练器（交叉熵损失）
- DPOTrainer: 直接偏好优化训练器（对比学习）
- GRPOTrainer: 组相对策略优化训练器（强化学习）

使用示例：
    >>> from src.trainer import SFTTrainer
    >>> from src.trainer.optimizer import configure_optimizer
    >>> 
    >>> optimizer = configure_optimizer(model, 'adamw', lr=1e-4)
    >>> trainer = SFTTrainer(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     device='cuda',
    ...     max_epochs=10,
    ...     use_amp=True
    ... )
    >>> trainer.train()

分布式训练：
    训练器自动检测分布式环境（torchrun），无需额外配置。
    
    单机4卡训练：
    >>> # torchrun --nproc_per_node=4 train.py
    >>> trainer = SFTTrainer(...)  # 自动启用 DDP
    >>> trainer.train()

特性：
- ✅ 分布式训练支持 (DDP)
- ✅ 混合精度训练 (AMP)
- ✅ 梯度累积
- ✅ 梯度裁剪
- ✅ 学习率调度
- ✅ 检查点管理
- ✅ 多种训练范式

作者：AnotherLLMFromScratch 项目
"""

from .base_trainer import BaseTrainer
from .sft_trainer import SFTTrainer, PretrainTrainer
from .dpo_trainer import DPOTrainer
from .grpo_trainer import GRPOTrainer

# 导出优化器工具
from .optimizer import (
    create_optimizer,
    configure_optimizer,
    get_parameter_groups
)

# 导出检查点工具（如果有）
try:
    from .checkpoint import (
        save_checkpoint,
        load_checkpoint,
        CheckpointManager
    )
    _has_checkpoint = True
except ImportError:
    _has_checkpoint = False

__all__ = [
    # 训练器
    'BaseTrainer',
    'SFTTrainer',
    'PretrainTrainer',
    'DPOTrainer',
    'GRPOTrainer',
    
    # 优化器工具
    'create_optimizer',
    'configure_optimizer',
    'get_parameter_groups',
]

# 如果有检查点模块，添加到导出列表
if _has_checkpoint:
    __all__.extend([
        'save_checkpoint',
        'load_checkpoint',
        'CheckpointManager'
    ])

__version__ = '1.0.0'
__author__ = 'AnotherLLMFromScratch'


# ==================== 训练器选择辅助函数 ====================

def get_trainer_class(trainer_type: str):
    """
    根据训练类型获取对应的训练器类
    
    参数：
        trainer_type: 训练类型 ('sft', 'dpo', 'grpo', 'pretrain')
    
    返回：
        对应的训练器类
    
    示例：
        >>> TrainerClass = get_trainer_class('sft')
        >>> trainer = TrainerClass(model=model, optimizer=optimizer, ...)
    """
    trainer_type = trainer_type.lower()
    
    trainer_map = {
        'sft': SFTTrainer,
        'pretrain': PretrainTrainer,
        'dpo': DPOTrainer,
        'grpo': GRPOTrainer,
    }
    
    if trainer_type not in trainer_map:
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available types: {list(trainer_map.keys())}"
        )
    
    return trainer_map[trainer_type]


def create_trainer(trainer_type: str, **kwargs):
    """
    创建训练器的工厂函数
    
    参数：
        trainer_type: 训练类型 ('sft', 'dpo', 'grpo', 'pretrain')
        **kwargs: 训练器参数
    
    返回：
        配置好的训练器实例
    
    示例：
        >>> trainer = create_trainer(
        ...     'sft',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     train_loader=train_loader,
        ...     max_epochs=10
        ... )
        >>> trainer.train()
    """
    TrainerClass = get_trainer_class(trainer_type)
    return TrainerClass(**kwargs)


# ==================== 模块信息 ====================

def print_trainer_info():
    """打印训练器模块信息"""
    info = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                      Trainer 模块信息                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  版本: {__version__}                                                    ║
║  作者: {__author__}                               ║
║                                                                   ║
║  可用训练器:                                                       ║
║  • SFTTrainer       - 监督微调（交叉熵损失）                      ║
║  • PretrainTrainer  - 预训练（与 SFT 相同）                       ║
║  • DPOTrainer       - 直接偏好优化（对比学习）                    ║
║  • GRPOTrainer      - 组相对策略优化（强化学习）                  ║
║                                                                   ║
║  特性:                                                             ║
║  ✅ 分布式训练 (DDP)                                               ║
║  ✅ 混合精度 (AMP)                                                 ║
║  ✅ 梯度累积                                                       ║
║  ✅ 梯度裁剪                                                       ║
║  ✅ 学习率调度                                                     ║
║  ✅ 检查点管理                                                     ║
║                                                                   ║
║  使用示例:                                                         ║
║  >>> from src.trainer import SFTTrainer                           ║
║  >>> trainer = SFTTrainer(model, optimizer, train_loader)         ║
║  >>> trainer.train()                                              ║
║                                                                   ║
║  分布式训练:                                                       ║
║  >>> # torchrun --nproc_per_node=4 train.py                       ║
║  >>> trainer = SFTTrainer(...)  # 自动检测并启用 DDP              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(info)


if __name__ == "__main__":
    print_trainer_info()

