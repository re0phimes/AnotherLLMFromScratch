"""
基础训练器 - Base Trainer
================================

所有训练器的基类，包含通用的训练逻辑。

特性：
- 分布式训练支持 (DDP)
- 混合精度训练 (AMP)
- 梯度累积
- 梯度裁剪
- 学习率调度
- 检查点管理

子类需要实现的方法：
- _prepare_batch(): 准备批次数据
- _forward(): 前向传播
- _compute_loss(): 计算损失

作者：AnotherLLMFromScratch 项目
"""

import time
import math
from typing import Optional, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod

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


class BaseTrainer(ABC):
    """
    基础训练器（抽象基类）
    
    所有具体训练器的父类，提供通用的训练逻辑框架。
    子类需要实现任务特定的方法。
    
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
        save_interval: int = 1,
        **kwargs  # 子类可能需要额外参数
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
        
        # 保存额外的kwargs供子类使用
        self.extra_config = kwargs
        
        if self.is_main:
            self._print_config()
    
    def _print_config(self):
        """打印训练配置（仅主进程）"""
        print(f"\n{'='*70}")
        print(f"{self.__class__.__name__} 配置")
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
        
        # 子类可以添加额外的配置信息
        extra_info = self._get_extra_config_info()
        if extra_info:
            for key, value in extra_info.items():
                print(f"{key}: {value}")
        
        print(f"{'='*70}\n")
    
    def _get_extra_config_info(self) -> Dict[str, Any]:
        """子类可以重写此方法来添加额外的配置信息"""
        return {}
    
    # ==================== 抽象方法：子类必须实现 ====================
    
    @abstractmethod
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备批次数据（子类必须实现）
        
        将原始batch转换为模型可以接受的格式，并移动到正确的设备。
        
        参数：
            batch: 原始batch字典
        
        返回：
            处理后的batch字典
        """
        pass
    
    @abstractmethod
    def _forward(self, batch: Dict[str, Any]) -> Any:
        """
        前向传播（子类必须实现）
        
        执行模型的前向传播，返回模型输出。
        
        参数：
            batch: 准备好的batch字典
        
        返回：
            模型输出（格式由子类决定）
        """
        pass
    
    @abstractmethod
    def _compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算损失（子类必须实现）
        
        根据模型输出和batch计算损失。
        
        参数：
            outputs: 模型输出
            batch: 准备好的batch字典
        
        返回：
            标量损失张量
        """
        pass
    
    # ==================== 通用训练逻辑 ====================
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 准备数据（子类实现）
            batch_prepared = self._prepare_batch(batch)
            
            # 前向传播（子类实现）
            with autocast('cuda', enabled=self.use_amp):
                outputs = self._forward(batch_prepared)
                loss = self._compute_loss(outputs, batch_prepared)
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
                    self._log_training_step(loss, batch_idx, batch_prepared)
            
            total_loss += loss.item() * self.grad_accum_steps
            total_tokens += self._count_tokens(batch_prepared)
        
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
            'tokens_per_sec': total_tokens / elapsed if elapsed > 0 else 0
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
            batch_prepared = self._prepare_batch(batch)
            
            with autocast('cuda', enabled=self.use_amp):
                outputs = self._forward(batch_prepared)
                loss = self._compute_loss(outputs, batch_prepared)
            
            total_loss += loss.item()
            total_tokens += self._count_tokens(batch_prepared)
        
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
    
    def _count_tokens(self, batch: Dict[str, Any]) -> int:
        """
        计算batch中的token数（子类可以重写）
        
        默认行为：查找 'input_ids' 键并返回其元素数量
        """
        if 'input_ids' in batch:
            return batch['input_ids'].numel()
        return 0
    
    def _log_training_step(self, loss: torch.Tensor, batch_idx: int, batch: Dict[str, Any]):
        """
        打印训练步骤日志（仅主进程）
        
        子类可以重写此方法来添加额外的日志信息
        """
        if not self.is_main:
            return
        
        current_loss = loss.item() * self.grad_accum_steps
        current_ppl = math.exp(min(current_loss, 20))
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 基础日志
        log_str = (f"Epoch {self.current_epoch} | "
                   f"Step {self.global_step} | "
                   f"Batch {batch_idx}/{len(self.train_loader)} | "
                   f"Loss: {current_loss:.4f} | "
                   f"PPL: {current_ppl:.2f} | "
                   f"LR: {current_lr:.6f}")
        
        # 子类可以添加额外信息
        extra_log = self._get_extra_log_info(batch)
        if extra_log:
            log_str += " | " + extra_log
        
        print(log_str)
    
    def _get_extra_log_info(self, batch: Dict[str, Any]) -> str:
        """子类可以重写此方法来添加额外的日志信息"""
        return ""
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """保存检查点（仅主进程）"""
        if not self.is_main:
            return
        
        # 获取原始模型（如果使用了 DDP）
        model_to_save = self.model.module if self.use_ddp else self.model
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 子类可以添加额外的状态
        extra_state = self._get_extra_checkpoint_state()
        if extra_state:
            checkpoint['extra_state'] = extra_state
        
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
        
        # 加载到原始模型（如果使用了 DDP）
        model_to_load = self.model.module if self.use_ddp else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 子类可以加载额外的状态
        if 'extra_state' in checkpoint:
            self._load_extra_checkpoint_state(checkpoint['extra_state'])
        
        if self.is_main:
            print(f"检查点已加载: {checkpoint_path}")
            print(f"恢复到 Epoch {self.current_epoch}, Step {self.global_step}")
    
    def _get_extra_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """子类可以重写此方法来保存额外的状态"""
        return None
    
    def _load_extra_checkpoint_state(self, extra_state: Dict[str, Any]):
        """子类可以重写此方法来加载额外的状态"""
        pass
    
    def train(self):
        """完整训练流程"""
        if self.is_main:
            print(f"\n{'='*70}")
            print(f"开始训练 - {self.__class__.__name__}")
            print(f"{'='*70}\n")
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            if self.is_main:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch + 1}/{self.max_epochs}")
                print(f"{'='*70}\n")
            
            # 训练
            train_stats = self.train_epoch()
            
            if self.is_main:
                print(f"\n训练统计:")
                print(f"  平均损失: {train_stats['loss']:.4f}")
                print(f"  困惑度: {train_stats['ppl']:.2f}")
                print(f"  耗时: {train_stats['time']:.2f}s")
                print(f"  吞吐量: {train_stats['tokens_per_sec']:.0f} tokens/s")
            
            # 验证
            if self.val_loader is not None:
                val_stats = self.validate()
                
                if self.is_main:
                    print(f"\n验证统计:")
                    print(f"  验证损失: {val_stats['loss']:.4f}")
                    print(f"  困惑度: {val_stats['ppl']:.2f}")
                
                is_best = val_stats['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_stats['loss']
                    if self.is_main:
                        print(f"  ✓ 新的最佳验证损失！")
            else:
                is_best = False
            
            # 保存检查点（仅主进程）
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(is_best=is_best)
        
        if self.is_main:
            print(f"\n{'='*70}")
            print("训练完成！")
            print(f"最佳验证损失: {self.best_val_loss:.4f}")
            print(f"{'='*70}\n")

