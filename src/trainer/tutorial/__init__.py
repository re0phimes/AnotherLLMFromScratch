"""
训练器教学模块
==================

本模块包含从零手搓的优化器和训练器实现，用于教学目的。
这些实现包含详细的注释，帮助理解深度学习训练的核心概念。

使用示例：
    from src.trainer.tutorial import AdamWFromScratch, TrainerFromScratch
    
    # 使用手搓的优化器
    optimizer = AdamWFromScratch(model.parameters(), lr=1e-3)
    
    # 使用手搓的训练器
    trainer = TrainerFromScratch(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader
    )
    trainer.train()

推荐学习路径：
1. 先阅读 optimizer_from_scratch.py 理解优化器原理
2. 再阅读 trainer_from_scratch.py 理解训练循环
3. 最后对比父目录的精简版本，理解如何使用 PyTorch 内置工具

作者：AnotherLLMFromScratch 项目
"""

from .optimizer_from_scratch import AdamWFromScratch, SGDFromScratch
from .trainer_from_scratch import TrainerFromScratch
from .train_with_custom_data import main as run_tutorial_training

__all__ = [
    'AdamWFromScratch',
    'SGDFromScratch',
    'TrainerFromScratch',
    'run_tutorial_training',
]

__version__ = '1.0.0'
__doc_url__ = 'https://github.com/AnotherLLMFromScratch'

# 教学提示
LEARNING_TIPS = """
╔═══════════════════════════════════════════════════════════════════╗
║                    教学模块使用指南                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  本模块包含从零实现的优化器和训练器，适合学习深度学习原理。          ║
║                                                                   ║
║  📚 推荐学习路径：                                                  ║
║  1. optimizer_from_scratch.py - 理解优化算法（AdamW, SGD）         ║
║  2. trainer_from_scratch.py   - 理解训练循环流程                   ║
║  3. ../optimizer.py           - 学习如何使用 PyTorch 优化器        ║
║  4. ../trainer.py             - 学习高效的训练器实现                ║
║                                                                   ║
║  💡 核心概念：                                                      ║
║  • 梯度下降与优化算法                                               ║
║  • 动量与自适应学习率                                               ║
║  • 梯度累积与梯度裁剪                                               ║
║  • 混合精度训练                                                     ║
║  • 学习率调度                                                       ║
║                                                                   ║
║  ⚠️  注意：                                                         ║
║  教学版本注重可读性，实际训练请使用父目录的精简版本以获得更好性能。   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def print_learning_tips():
    """打印学习提示"""
    print(LEARNING_TIPS)


# 自动打印学习提示（可选）
if __name__ == "__main__":
    print_learning_tips()

