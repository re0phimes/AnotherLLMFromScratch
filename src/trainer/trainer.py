"""
训练器模块 - 实际训练版本
================================

精简高效的训练器实现，使用 PyTorch 内置功能。
适合实际项目使用。

如果想了解训练循环的详细原理，请查看 tutorial/trainer_from_scratch.py

主要特性：
- 混合精度训练 (AMP)
- 梯度累积
- 梯度裁剪
- 学习率调度
- 检查点管理
- 分布式训练支持 (DDP)

作者：AnotherLLMFromScratch 项目
"""

import time
import math
from typing import Optional, Dict, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# 导入分布式工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.distributed import (
    is_distributed, 
    is_main_process, 
    get_rank,
    get_local_rank,
    get_world_size,
    reduce_tensor,
    setup_distributed,
    cleanup_distributed,
    get_device
)


class Trainer:
    """
    高效的训练器实现，支持单机/分布式训练
    
    参数：
        model: 模型
        optimizer: 优化器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        scheduler: 学习率调度器（可选）
        device: 训练设备（如果为 None 则自动检测）
        max_epochs: 最大训练轮数
        grad_accum_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值（0 表示不裁剪）
        use_amp: 是否使用混合精度
        use_ddp: 是否使用 DistributedDataParallel（自动检测）
        log_interval: 日志打印间隔
        save_dir: 检查点保存目录
        save_interval: 检查点保存间隔（按 epoch）
    
    使用示例：
        单机训练：
        >>> trainer = Trainer(model, optimizer, train_loader)
        >>> trainer.train()
        
        分布式训练（使用 torchrun）：
        >>> # torchrun --nproc_per_node=4 train.py
        >>> trainer = Trainer(model, optimizer, train_loader)
        >>> trainer.train()  # 自动检测并启用 DDP
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None,
        max_epochs: int = 10,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        use_ddp: Optional[bool] = None,
        log_interval: int = 10,
        save_dir: str = './checkpoints',
        save_interval: int = 1
    ):
        # 检测分布式环境
        self.is_distributed = is_distributed()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()
        
        # 自动确定设备
        if device is None:
            device = get_device()
        self.device = device
        
        # 移动模型到设备
        self.model = model.to(self.device)
        
        # 自动决定是否使用 DDP
        if use_ddp is None:
            use_ddp = self.is_distributed
        self.use_ddp = use_ddp
        
        # 如果是分布式训练，使用 DDP 包装模型
        if self.use_ddp and self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None
            )
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        
        self.max_epochs = max_epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and 'cuda' in self.device
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        self.save_dir = Path(save_dir)
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        if self.is_main:
            self._print_config()
    
    def _print_config(self):
        """打印训练配置（仅主进程）"""
        print(f"\n{'='*70}")
        print("训练器配置")
        print(f"{'='*70}")
        print(f"分布式训练: {self.is_distributed}")
        if self.is_distributed:
            print(f"Rank: {self.rank}/{self.world_size}")
            print(f"Local Rank: {self.local_rank}")
            print(f"DDP启用: {self.use_ddp}")
        print(f"设备: {self.device}")
        print(f"最大轮数: {self.max_epochs}")
        print(f"梯度累积: {self.grad_accum_steps}")
        print(f"梯度裁剪: {self.max_grad_norm}")
        print(f"混合精度: {self.use_amp}")
        print(f"学习率调度器: {type(self.scheduler).__name__ if self.scheduler else 'None'}")
        print(f"{'='*70}\n")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 准备数据
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            # 前向传播
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs, labels)
                loss = loss / self.grad_accum_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积后更新
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # 优化器步进
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
                
                # 打印日志
                if self.global_step % self.log_interval == 0:
                    self._log_training_step(loss, batch_idx)
            
            total_loss += loss.item() * self.grad_accum_steps
            total_tokens += input_ids.numel()
        
        avg_loss = total_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        
        # 分布式训练：同步损失和 token 数
        if self.is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)
            avg_loss_tensor = reduce_tensor(avg_loss_tensor, op='mean')
            total_tokens_tensor = reduce_tensor(total_tokens_tensor, op='sum')
            avg_loss = avg_loss_tensor.item()
            total_tokens = int(total_tokens_tensor.item())
        
        return {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20)),  # 限制防止溢出
            'time': elapsed,
            'tokens_per_sec': total_tokens / elapsed
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs, labels)
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # 分布式训练：同步验证损失
        if self.is_distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss_tensor = reduce_tensor(avg_loss_tensor, op='mean')
            avg_loss = avg_loss_tensor.item()
        
        return {
            'loss': avg_loss,
            'ppl': math.exp(min(avg_loss, 20))
        }
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算交叉熵损失"""
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
    
    def _log_training_step(self, loss: float, batch_idx: int):
        """打印训练步骤日志"""
        current_loss = loss.item() * self.grad_accum_steps
        current_ppl = math.exp(min(current_loss, 20))
        current_lr = self.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {self.current_epoch} | "
              f"Step {self.global_step} | "
              f"Batch {batch_idx}/{len(self.train_loader)} | "
              f"Loss: {current_loss:.4f} | "
              f"PPL: {current_ppl:.2f} | "
              f"LR: {current_lr:.6f}")
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存常规检查点
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pt'
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"检查点已加载: {checkpoint_path}")
        print(f"恢复到 Epoch {self.current_epoch}, Step {self.global_step}")
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*70}")
        print("开始训练")
        print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"{'='*70}\n")
            
            # 训练
            train_stats = self.train_epoch()
            print(f"\n训练统计:")
            print(f"  平均损失: {train_stats['loss']:.4f}")
            print(f"  困惑度: {train_stats['ppl']:.2f}")
            print(f"  耗时: {train_stats['time']:.2f}s")
            print(f"  吞吐量: {train_stats['tokens_per_sec']:.0f} tokens/s")
            
            # 验证
            if self.val_loader is not None:
                val_stats = self.validate()
                print(f"\n验证统计:")
                print(f"  验证损失: {val_stats['loss']:.4f}")
                print(f"  困惑度: {val_stats['ppl']:.2f}")
                
                is_best = val_stats['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_stats['loss']
                    print(f"  ✓ 新的最佳验证损失！")
            else:
                is_best = False
            
            # 保存检查点
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(is_best=is_best)
        
        print(f"\n{'='*70}")
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """演示训练器的使用"""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    print("=" * 70)
    print("训练器使用示例")
    print("=" * 70)
    
    # 创建模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.fc = nn.Linear(256, 1000)
        
        def forward(self, x):
            return self.fc(self.embedding(x))
    
    model = DummyModel()
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # 创建调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    # 创建数据加载器（虚拟数据）
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 50
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (32,)),
                'labels': torch.randint(0, 1000, (32,))
            }
    
    train_loader = DataLoader(DummyDataset(), batch_size=4)
    val_loader = DataLoader(DummyDataset(), batch_size=4)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device='cpu',
        max_epochs=2,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        use_amp=False,
        log_interval=5
    )
    
    # 开始训练
    trainer.train()
    
    print("\n提示：查看 tutorial/trainer_from_scratch.py 了解训练循环的详细原理")
