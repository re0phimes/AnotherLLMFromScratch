"""
手搓训练器 - 教学版本
========================

本模块从零实现一个完整的训练器，用于教学目的。
包含详细的训练循环、验证循环、梯度裁剪、学习率调度等核心概念的注释。

训练循环的核心步骤：
1. 前向传播 (Forward Pass): 计算模型输出和损失
2. 反向传播 (Backward Pass): 计算梯度
3. 梯度裁剪 (Gradient Clipping): 防止梯度爆炸
4. 优化器更新 (Optimizer Step): 更新模型参数
5. 学习率调度 (Learning Rate Schedule): 动态调整学习率

作者：AnotherLLMFromScratch 项目
用途：教学演示
"""

import time
import math
from typing import Optional, Dict, List, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast


class TrainerFromScratch:
    """
    从零实现的训练器
    
    这个训练器实现了标准的深度学习训练流程，包括：
    - 训练循环和验证循环
    - 梯度累积（模拟更大的 batch size）
    - 梯度裁剪（防止梯度爆炸）
    - 混合精度训练（加速训练并节省显存）
    - 学习率预热和衰减
    - 检查点保存和恢复
    
    参数说明：
        model: 要训练的神经网络模型
        optimizer: 优化器（可以是手搓的或 PyTorch 的）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        device: 训练设备（'cuda' 或 'cpu'）
        max_epochs: 最大训练轮数
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪的最大范数
        use_amp: 是否使用混合精度训练
        warmup_steps: 学习率预热步数
        log_interval: 日志打印间隔
        save_dir: 检查点保存目录
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: object,  # 可以是任何优化器（手搓的或 PyTorch 的）
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        warmup_steps: int = 0,
        log_interval: int = 10,
        save_dir: str = './checkpoints'
    ):
        # 基本组件
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 训练配置
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps  # 梯度累积步数
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值
        self.use_amp = use_amp and device == 'cuda'  # 混合精度仅在 GPU 上启用
        self.warmup_steps = warmup_steps  # 预热步数
        self.log_interval = log_interval  # 日志间隔
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0  # 全局训练步数
        self.best_val_loss = float('inf')
        
        # 混合精度训练的缩放器
        # GradScaler 用于自动缩放损失，防止混合精度训练时的数值下溢
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # 学习率调度器（可选，这里使用余弦退火）
        self.base_lr = optimizer.lr if hasattr(optimizer, 'lr') else 1e-3
        
        print(f"训练器初始化完成:")
        print(f"  设备: {device}")
        print(f"  混合精度: {self.use_amp}")
        print(f"  梯度累积步数: {gradient_accumulation_steps}")
        print(f"  梯度裁剪阈值: {max_grad_norm}")
    
    def get_lr(self, step: int) -> float:
        """
        计算当前步的学习率（带预热和余弦衰减）
        
        学习率调度策略：
        1. Warmup 阶段（0 到 warmup_steps）：线性增长
           lr = base_lr * (step / warmup_steps)
        
        2. 余弦衰减阶段（warmup_steps 之后）：
           lr = 0.1 * base_lr + 0.5 * base_lr * (1 + cos(π * progress))
           其中 progress = (step - warmup) / (total_steps - warmup)
        
        参数：
            step: 当前训练步数
        
        返回：
            当前步的学习率
        """
        # 总训练步数
        total_steps = len(self.train_loader) * self.max_epochs
        
        # 预热阶段：线性增长
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        
        # 余弦衰减阶段
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        # 余弦退火：从 1.0 衰减到 0.1
        cosine_decay = 0.1 + 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * cosine_decay
    
    def update_lr(self, lr: float):
        """
        更新优化器的学习率
        
        参数：
            lr: 新的学习率
        """
        if hasattr(self.optimizer, 'set_lr'):
            # 使用手搓优化器的方法
            self.optimizer.set_lr(lr)
        elif hasattr(self.optimizer, 'param_groups'):
            # 使用 PyTorch 优化器的方法
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def clip_gradients(self):
        """
        梯度裁剪 (Gradient Clipping)
        
        为什么需要梯度裁剪？
        - 防止梯度爆炸（Gradient Explosion）
        - 稳定训练过程，特别是在训练 RNN 和 Transformer 时
        
        裁剪方法：
        - 计算所有参数梯度的总范数 ||g||
        - 如果 ||g|| > max_grad_norm，缩放梯度：g = g * (max_grad_norm / ||g||)
        
        返回：
            梯度的总范数
        """
        if self.max_grad_norm <= 0:
            return 0.0
        
        # 收集所有参数的梯度
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # 计算梯度的总范数
        # ||g|| = sqrt(sum(||g_i||²))
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in parameters])
        ).item()
        
        # 计算裁剪系数
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        
        # 如果梯度范数超过阈值，进行裁剪
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
        
        return total_norm
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个 epoch
        
        训练循环的核心逻辑：
        1. 遍历训练数据
        2. 前向传播计算损失
        3. 反向传播计算梯度（使用梯度累积）
        4. 裁剪梯度
        5. 更新参数
        6. 调度学习率
        
        返回：
            包含训练统计信息的字典
        """
        self.model.train()  # 设置为训练模式（启用 dropout、batch norm 等）
        
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        # 遍历训练数据
        for batch_idx, batch in enumerate(self.train_loader):
            # ==================== 步骤 1: 准备数据 ====================
            # 将数据移到设备上（GPU 或 CPU）
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # ==================== 步骤 2: 前向传播 ====================
            # 使用混合精度训练（如果启用）
            if self.use_amp:
                # autocast 自动将某些操作转换为 float16，加速训练
                with autocast('cuda'):
                    outputs = self.model(input_ids)
                    loss = self._compute_loss(outputs, labels)
            else:
                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs, labels)
            
            # 梯度累积：将损失除以累积步数
            # 这样累积后的梯度相当于更大 batch size 的梯度
            loss = loss / self.gradient_accumulation_steps
            
            # ==================== 步骤 3: 反向传播 ====================
            if self.use_amp:
                # 使用 GradScaler 缩放损失，防止混合精度训练时的下溢
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # ==================== 步骤 4: 梯度累积和参数更新 ====================
            # 只在累积足够的梯度后才更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 4.1 梯度裁剪
                if self.use_amp:
                    # 在混合精度训练中，需要先 unscale 梯度
                    self.scaler.unscale_(self.optimizer)
                grad_norm = self.clip_gradients()
                
                # 4.2 优化器步进
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # 4.3 清零梯度
                self.optimizer.zero_grad()
                
                # 4.4 更新学习率
                self.global_step += 1
                new_lr = self.get_lr(self.global_step)
                self.update_lr(new_lr)
                
                # ==================== 步骤 5: 记录日志 ====================
                if self.global_step % self.log_interval == 0:
                    # 计算困惑度 (Perplexity)
                    # PPL = exp(loss)，衡量语言模型的质量
                    ppl = math.exp(loss.item() * self.gradient_accumulation_steps)
                    
                    # 计算吞吐量（tokens/second）
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    
                    print(f"Epoch {self.current_epoch} | "
                          f"Step {self.global_step} | "
                          f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} | "
                          f"PPL: {ppl:.2f} | "
                          f"LR: {new_lr:.6f} | "
                          f"Grad Norm: {grad_norm:.4f} | "
                          f"Tokens/s: {tokens_per_sec:.0f}")
            
            # 更新统计信息
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_tokens += input_ids.numel()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        elapsed_time = time.time() - start_time
        
        return {
            'loss': avg_loss,
            'ppl': math.exp(avg_loss),
            'time': elapsed_time,
            'tokens_per_sec': total_tokens / elapsed_time
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        验证模型性能
        
        验证循环：
        1. 设置模型为评估模式
        2. 禁用梯度计算（节省内存和计算）
        3. 遍历验证数据计算损失
        
        返回：
            包含验证统计信息的字典
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()  # 设置为评估模式（禁用 dropout、batch norm 等）
        
        total_loss = 0.0
        total_tokens = 0
        
        # 遍历验证数据
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播（不需要梯度）
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(input_ids)
                    loss = self._compute_loss(outputs, labels)
            else:
                outputs = self.model(input_ids)
                loss = self._compute_loss(outputs, labels)
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'ppl': math.exp(avg_loss)
        }
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失
        
        对于语言模型：
        - logits: [batch_size, seq_len, vocab_size] 模型输出
        - labels: [batch_size, seq_len] 真实标签
        
        参数：
            logits: 模型输出的 logits
            labels: 真实标签
        
        返回：
            标量损失值
        """
        # 展平 logits 和 labels
        # logits: [batch_size * seq_len, vocab_size]
        # labels: [batch_size * seq_len]
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # 计算交叉熵损失
        # ignore_index=-100: 忽略填充 token 的损失
        loss = nn.functional.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100
        )
        
        return loss
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存检查点
        
        检查点包含：
        - 模型参数
        - 优化器状态
        - 训练进度
        - 最佳验证损失
        
        参数：
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存优化器状态
        if hasattr(self.optimizer, 'state_dict'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # 保存 GradScaler 状态（混合精度训练）
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
    
    def train(self):
        """
        完整的训练流程
        
        训练流程：
        1. 训练一个 epoch
        2. 验证模型（如果有验证集）
        3. 保存检查点
        4. 重复直到达到最大 epoch 数
        """
        print("\n" + "=" * 70)
        print("开始训练")
        print("=" * 70)
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"{'='*70}")
            
            # 训练一个 epoch
            train_stats = self.train_epoch()
            print(f"\n训练统计:")
            print(f"  平均损失: {train_stats['loss']:.4f}")
            print(f"  困惑度: {train_stats['ppl']:.2f}")
            print(f"  耗时: {train_stats['time']:.2f}s")
            print(f"  吞吐量: {train_stats['tokens_per_sec']:.0f} tokens/s")
            
            # 验证模型
            if self.val_loader is not None:
                val_stats = self.validate()
                print(f"\n验证统计:")
                print(f"  验证损失: {val_stats['loss']:.4f}")
                print(f"  困惑度: {val_stats['ppl']:.2f}")
                
                # 检查是否是最佳模型
                is_best = val_stats['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_stats['loss']
                    print(f"  ✓ 新的最佳验证损失！")
                
                # 保存检查点
                self.save_checkpoint(is_best=is_best)
            else:
                # 没有验证集，直接保存
                self.save_checkpoint(is_best=False)
        
        print("\n" + "=" * 70)
        print("训练完成！")
        print("=" * 70)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """
    演示如何使用手搓的训练器
    """
    print("=" * 70)
    print("手搓训练器教学演示")
    print("=" * 70)
    
    # 创建一个简单的模型（用于演示）
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.fc = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            logits = self.fc(x)
            return logits
    
    model = DummyModel()
    
    # 创建优化器（可以使用手搓的或 PyTorch 的）
    from optimizer_from_scratch import AdamWFromScratch
    optimizer = AdamWFromScratch(
        params=list(model.parameters()),
        lr=1e-3,
        weight_decay=0.01
    )
    
    # 创建虚拟数据加载器（用于演示）
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (32,)),
                'labels': torch.randint(0, 1000, (32,))
            }
    
    train_loader = DataLoader(DummyDataset(), batch_size=4, shuffle=True)
    val_loader = DataLoader(DummyDataset(), batch_size=4, shuffle=False)
    
    # 创建训练器
    trainer = TrainerFromScratch(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',  # 演示用 CPU
        max_epochs=2,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        use_amp=False,
        warmup_steps=10,
        log_interval=5
    )
    
    # 开始训练
    trainer.train()
    
    print("\n演示完成！")

