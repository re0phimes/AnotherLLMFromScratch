"""
SFT 训练器 - Supervised Fine-Tuning Trainer
==============================================

用于指令微调（Supervised Fine-Tuning）的训练器。

训练方式：
- 使用交叉熵损失
- prompt 部分的 label 通常设为 -100（不计算损失）
- 只对 response 部分计算损失

数据格式：
输入 batch 应包含：
- input_ids: [batch_size, seq_len] - 输入序列（prompt + response）
- labels: [batch_size, seq_len] - 标签（prompt部分为-100）
- attention_mask: [batch_size, seq_len] - 注意力掩码（可选）

使用示例：
    >>> from src.trainer.sft_trainer import SFTTrainer
    >>> from src.trainer.optimizer import configure_optimizer
    >>> 
    >>> optimizer = configure_optimizer(model, 'adamw', lr=1e-4)
    >>> trainer = SFTTrainer(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader
    ... )
    >>> trainer.train()

作者：AnotherLLMFromScratch 项目
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    """
    SFT（监督微调）训练器
    
    使用标准的交叉熵损失训练语言模型。
    适用于指令微调和预训练任务。
    
    参数：
        与 BaseTrainer 相同
    
    示例：
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     train_loader=train_loader,
        ...     max_epochs=3,
        ...     use_amp=True
        ... )
        >>> trainer.train()
    """
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备 SFT 训练的 batch
        
        将数据移动到正确的设备，并确保格式正确。
        
        参数：
            batch: 原始 batch，应包含 'input_ids' 和 'labels'
        
        返回：
            处理后的 batch 字典
        """
        prepared = {}
        
        # 移动 input_ids 到设备
        if 'input_ids' in batch:
            prepared['input_ids'] = batch['input_ids'].to(self.device)
        else:
            raise ValueError("Batch must contain 'input_ids'")
        
        # 移动 labels 到设备
        # 如果没有 labels，使用 input_ids（用于预训练）
        if 'labels' in batch:
            prepared['labels'] = batch['labels'].to(self.device)
        else:
            prepared['labels'] = prepared['input_ids'].clone()
        
        # 移动 attention_mask 到设备（如果存在）
        if 'attention_mask' in batch:
            prepared['attention_mask'] = batch['attention_mask'].to(self.device)
        
        return prepared
    
    def _forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        SFT 的前向传播
        
        执行单次前向传播，返回模型的 logits。
        
        参数：
            batch: 准备好的 batch，包含 'input_ids' 和可选的 'attention_mask'
        
        返回：
            logits: [batch_size, seq_len, vocab_size] 模型输出
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # 前向传播
        if attention_mask is not None:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids)
        
        # 处理不同的模型输出格式
        if isinstance(outputs, torch.Tensor):
            # 模型直接返回 logits
            logits = outputs
        elif hasattr(outputs, 'logits'):
            # 模型返回带 logits 属性的对象（如 HuggingFace 模型）
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            # 模型返回字典
            logits = outputs['logits']
        else:
            raise ValueError(f"Unsupported model output format: {type(outputs)}")
        
        return logits
    
    def _compute_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        计算 SFT 的交叉熵损失
        
        参数：
            outputs: 模型输出的 logits [batch_size, seq_len, vocab_size]
            batch: 包含 'labels' 的 batch 字典
        
        返回：
            标量损失值
        """
        logits = outputs
        labels = batch['labels']
        
        # 展平 logits 和 labels 以便计算交叉熵
        # logits: [batch_size * seq_len, vocab_size]
        # labels: [batch_size * seq_len]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算交叉熵损失
        # ignore_index=-100: 忽略 padding 和 prompt 部分的损失
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        return loss
    
    def _count_tokens(self, batch: Dict[str, Any]) -> int:
        """
        计算有效的 token 数量（不包括 padding）
        
        参数：
            batch: 包含 'labels' 的 batch 字典
        
        返回：
            有效 token 数量
        """
        labels = batch.get('labels', None)
        if labels is not None:
            # 只计算非 -100 的 token（即实际参与损失计算的 token）
            return (labels != -100).sum().item()
        else:
            # 如果没有 labels，使用 input_ids 的大小
            return batch['input_ids'].numel()
    
    def _get_extra_config_info(self) -> Dict[str, Any]:
        """提供 SFT 特定的配置信息"""
        return {
            '训练类型': 'SFT (Supervised Fine-Tuning)',
            '损失函数': 'Cross Entropy'
        }


# ==================== 便捷别名 ====================

# 预训练使用相同的损失函数，可以直接使用 SFTTrainer
PretrainTrainer = SFTTrainer


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """演示 SFT 训练器的使用"""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    
    print("=" * 70)
    print("SFT Trainer 使用示例")
    print("=" * 70)
    
    # 创建一个简单的模型
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=10000, d_model=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.fc = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            logits = self.fc(x)
            return logits
    
    model = DummyModel()
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 创建虚拟数据集
    class DummySFTDataset(Dataset):
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            seq_len = 128
            input_ids = torch.randint(0, 10000, (seq_len,))
            
            # 模拟 SFT 数据：前 32 个 token 是 prompt（label=-100），后面是 response
            labels = input_ids.clone()
            labels[:32] = -100  # prompt 部分不计算损失
            
            return {
                'input_ids': input_ids,
                'labels': labels
            }
    
    train_dataset = DummySFTDataset()
    val_dataset = DummySFTDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 创建 SFT 训练器
    trainer = SFTTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',  # 演示用 CPU
        max_epochs=2,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        use_amp=False,
        log_interval=5,
        save_dir='./checkpoints_sft'
    )
    
    print("\n开始训练...")
    trainer.train()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

