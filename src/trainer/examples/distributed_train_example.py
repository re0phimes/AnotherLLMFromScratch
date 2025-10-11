"""
分布式训练示例
==================

演示如何使用 Trainer 进行单机多卡或多机多卡的分布式训练。

运行方式：
    单机单卡（普通训练）：
        python distributed_train_example.py
    
    单机多卡（4卡）：
        torchrun --nproc_per_node=4 distributed_train_example.py
    
    多机多卡（2机8卡，每机4卡）：
        # 主节点（节点0）
        torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
                 --master_addr="192.168.1.1" --master_port=29500 \
                 distributed_train_example.py
        
        # 从节点（节点1）
        torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
                 --master_addr="192.168.1.1" --master_port=29500 \
                 distributed_train_example.py

作者：AnotherLLMFromScratch 项目
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.trainer.trainer import Trainer
from src.trainer.optimizer import configure_optimizer
from src.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    print_distributed_info
)


# ==================== 定义模型 ====================
class SimpleTransformer(nn.Module):
    """简单的 Transformer 模型用于演示"""
    
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(1024, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embedding + Position
        x = self.embedding(x) + self.pos_encoding(positions)
        
        # Transformer
        x = self.transformer(x)
        
        # 输出
        logits = self.fc_out(x)
        return logits


# ==================== 定义数据集 ====================
class DummyDataset(Dataset):
    """虚拟数据集用于演示"""
    
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=10000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机数据
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


# ==================== 主函数 ====================
def main():
    """主训练函数"""
    
    # ==================== 步骤 1: 初始化分布式环境 ====================
    if is_distributed():
        rank, local_rank, world_size = setup_distributed()
        print(f"[Rank {rank}] 分布式环境初始化完成")
    else:
        rank, local_rank, world_size = 0, 0, 1
        print("单进程模式")
    
    # 打印分布式信息（仅主进程）
    if is_main_process():
        print_distributed_info()
    
    # ==================== 步骤 2: 配置参数 ====================
    config = {
        'vocab_size': 10000,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_epochs': 10,
        'grad_accum_steps': 1,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'log_interval': 10,
        'save_interval': 1,
    }
    
    if is_main_process():
        print("\n" + "=" * 70)
        print("训练配置")
        print("=" * 70)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 70 + "\n")
    
    # ==================== 步骤 3: 创建数据集和数据加载器 ====================
    train_dataset = DummyDataset(num_samples=1000, seq_len=128)
    val_dataset = DummyDataset(num_samples=200, seq_len=128)
    
    # 分布式训练需要使用 DistributedSampler
    if is_distributed():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        shuffle = False  # sampler 已经处理 shuffle
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    if is_main_process():
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(val_dataset)}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}\n")
    
    # ==================== 步骤 4: 创建模型 ====================
    model = SimpleTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers']
    )
    
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}\n")
    
    # ==================== 步骤 5: 创建优化器 ====================
    optimizer = configure_optimizer(
        model,
        optimizer_type='adamw',
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        use_parameter_groups=True
    )
    
    # ==================== 步骤 6: 创建学习率调度器 ====================
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    total_steps = len(train_loader) * config['max_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # ==================== 步骤 7: 创建训练器 ====================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=None,  # 自动检测
        max_epochs=config['max_epochs'],
        grad_accum_steps=config['grad_accum_steps'],
        max_grad_norm=config['max_grad_norm'],
        use_amp=config['use_amp'],
        use_ddp=None,  # 自动检测
        log_interval=config['log_interval'],
        save_dir='./checkpoints',
        save_interval=config['save_interval']
    )
    
    # ==================== 步骤 8: 开始训练 ====================
    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main_process():
            print("\n训练被用户中断")
    except Exception as e:
        if is_main_process():
            print(f"\n训练出错: {e}")
        raise
    finally:
        # ==================== 步骤 9: 清理分布式环境 ====================
        if is_distributed():
            cleanup_distributed()
            if is_main_process():
                print("分布式环境已清理")


if __name__ == "__main__":
    main()

