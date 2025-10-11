"""
DPO 训练器 - Direct Preference Optimization Trainer
====================================================

用于直接偏好优化（Direct Preference Optimization）的训练器。

DPO 原理：
- 使用偏好对比数据（chosen vs rejected）
- 不需要显式的奖励模型
- 通过优化隐式奖励来对齐模型

DPO 损失公式：
    L_DPO = -log(σ(β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x))))
    
    其中：
    - y_w: chosen（更好的回答）
    - y_l: rejected（较差的回答）
    - π_θ: 当前策略模型
    - π_ref: 参考模型（冻结）
    - β: 温度参数
    - σ: sigmoid 函数

数据格式：
输入 batch 应包含（由 DPODatasetModule 生成）：
- prompt_input_ids: [batch_size, seq_len]
- chosen_input_ids: [batch_size, seq_len]
- rejected_input_ids: [batch_size, seq_len]
- prompt_attention_mask: [batch_size, seq_len]
- chosen_attention_mask: [batch_size, seq_len]
- rejected_attention_mask: [batch_size, seq_len]

使用示例：
    >>> from src.trainer.dpo_trainer import DPOTrainer
    >>> 
    >>> # 创建参考模型（通常是 SFT 后的模型副本）
    >>> ref_model = copy.deepcopy(model)
    >>> ref_model.eval()
    >>> 
    >>> trainer = DPOTrainer(
    ...     model=model,
    ...     ref_model=ref_model,
    ...     optimizer=optimizer,
    ...     train_loader=train_loader,
    ...     beta=0.1
    ... )
    >>> trainer.train()

作者：AnotherLLMFromScratch 项目
"""

from typing import Dict, Any, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer


class DPOTrainer(BaseTrainer):
    """
    DPO（直接偏好优化）训练器
    
    使用对比学习的方式训练模型，使其更倾向于 chosen 回答而非 rejected 回答。
    
    参数：
        model: 策略模型（要训练的模型）
        ref_model: 参考模型（冻结的模型，通常是 SFT 后的模型副本）
        optimizer: 优化器
        train_loader: 训练数据加载器（需要 DPO 格式数据）
        val_loader: 验证数据加载器
        beta: DPO 温度参数，控制偏好强度（默认 0.1）
        label_smoothing: 标签平滑系数（默认 0.0）
        **kwargs: 其他 BaseTrainer 参数
    
    示例：
        >>> import copy
        >>> ref_model = copy.deepcopy(model)
        >>> ref_model.eval()
        >>> 
        >>> trainer = DPOTrainer(
        ...     model=model,
        ...     ref_model=ref_model,
        ...     optimizer=optimizer,
        ...     train_loader=dpo_train_loader,
        ...     beta=0.1
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        optimizer,
        train_loader,
        val_loader=None,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        # 保存 DPO 特定参数
        self.beta = beta
        self.label_smoothing = label_smoothing
        
        # 初始化基类
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs
        )
        
        # 设置参考模型
        self.ref_model = ref_model
        self.ref_model.eval()  # 冻结参考模型
        
        # 将参考模型移到相同设备
        self.ref_model = self.ref_model.to(self.device)
        
        # 如果使用分布式，也需要包装参考模型（但不训练）
        if self.use_ddp and self.is_distributed:
            self.ref_model = nn.parallel.DistributedDataParallel(
                self.ref_model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None
            )
        
        # 确保参考模型的所有参数都不需要梯度
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备 DPO 训练的 batch
        
        DPO 需要 prompt、chosen 和 rejected 三部分数据。
        
        参数：
            batch: 原始 batch，应包含 DPO 格式的数据
        
        返回：
            处理后的 batch 字典
        """
        prepared = {}
        
        # 检查必需的字段
        required_keys = [
            'prompt_input_ids', 
            'chosen_input_ids', 
            'rejected_input_ids'
        ]
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"DPO batch must contain '{key}'")
        
        # 移动所有张量到设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        
        # 拼接 prompt + chosen 和 prompt + rejected
        # 注意：这里假设数据集已经正确处理了拼接
        # 如果数据集没有拼接，需要在这里拼接
        if 'full_chosen_input_ids' not in prepared:
            # 拼接 prompt + chosen
            prepared['full_chosen_input_ids'] = torch.cat([
                prepared['prompt_input_ids'],
                prepared['chosen_input_ids']
            ], dim=1)
            
            if 'chosen_attention_mask' in prepared and 'prompt_attention_mask' in prepared:
                prepared['full_chosen_attention_mask'] = torch.cat([
                    prepared['prompt_attention_mask'],
                    prepared['chosen_attention_mask']
                ], dim=1)
        
        if 'full_rejected_input_ids' not in prepared:
            # 拼接 prompt + rejected
            prepared['full_rejected_input_ids'] = torch.cat([
                prepared['prompt_input_ids'],
                prepared['rejected_input_ids']
            ], dim=1)
            
            if 'rejected_attention_mask' in prepared and 'prompt_attention_mask' in prepared:
                prepared['full_rejected_attention_mask'] = torch.cat([
                    prepared['prompt_attention_mask'],
                    prepared['rejected_attention_mask']
                ], dim=1)
        
        return prepared
    
    def _forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        DPO 的前向传播
        
        需要对 chosen 和 rejected 分别前向传播两次（策略模型和参考模型）。
        
        参数：
            batch: 准备好的 batch
        
        返回：
            包含四个 logits 的字典：
            - policy_chosen_logits: 策略模型对 chosen 的输出
            - policy_rejected_logits: 策略模型对 rejected 的输出
            - ref_chosen_logits: 参考模型对 chosen 的输出
            - ref_rejected_logits: 参考模型对 rejected 的输出
        """
        # 策略模型前向传播
        policy_chosen_logits = self._get_logits(
            self.model, 
            batch['full_chosen_input_ids'],
            batch.get('full_chosen_attention_mask', None)
        )
        
        policy_rejected_logits = self._get_logits(
            self.model,
            batch['full_rejected_input_ids'],
            batch.get('full_rejected_attention_mask', None)
        )
        
        # 参考模型前向传播（无梯度）
        with torch.no_grad():
            ref_chosen_logits = self._get_logits(
                self.ref_model,
                batch['full_chosen_input_ids'],
                batch.get('full_chosen_attention_mask', None)
            )
            
            ref_rejected_logits = self._get_logits(
                self.ref_model,
                batch['full_rejected_input_ids'],
                batch.get('full_rejected_attention_mask', None)
            )
        
        return {
            'policy_chosen_logits': policy_chosen_logits,
            'policy_rejected_logits': policy_rejected_logits,
            'ref_chosen_logits': ref_chosen_logits,
            'ref_rejected_logits': ref_rejected_logits
        }
    
    def _get_logits(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        获取模型的 logits
        
        参数：
            model: 模型
            input_ids: 输入序列
            attention_mask: 注意力掩码
        
        返回：
            logits
        """
        if attention_mask is not None:
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids)
        
        # 处理不同的模型输出格式
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            return outputs['logits']
        else:
            raise ValueError(f"Unsupported model output format: {type(outputs)}")
    
    def _compute_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算 DPO 损失
        
        DPO 损失基于对比学习，使模型更倾向于 chosen 而非 rejected。
        
        参数：
            outputs: 包含四个 logits 的字典
            batch: 包含标签信息的 batch
        
        返回：
            标量损失值
        """
        # 获取序列长度
        prompt_len = batch['prompt_input_ids'].size(1)
        
        # 计算 log 概率
        policy_chosen_logps = self._get_batch_logps(
            outputs['policy_chosen_logits'],
            batch['full_chosen_input_ids'],
            prompt_len
        )
        
        policy_rejected_logps = self._get_batch_logps(
            outputs['policy_rejected_logits'],
            batch['full_rejected_input_ids'],
            prompt_len
        )
        
        ref_chosen_logps = self._get_batch_logps(
            outputs['ref_chosen_logits'],
            batch['full_chosen_input_ids'],
            prompt_len
        )
        
        ref_rejected_logps = self._get_batch_logps(
            outputs['ref_rejected_logits'],
            batch['full_rejected_input_ids'],
            prompt_len
        )
        
        # 计算 log 比率（policy / reference）
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO 损失
        # losses = -log(sigmoid(beta * (policy_logratios - ref_logratios)))
        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))
        
        # 标签平滑（如果启用）
        if self.label_smoothing > 0:
            # 反向损失（偏好 rejected）
            reverse_losses = -F.logsigmoid(self.beta * (ref_logratios - policy_logratios))
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * reverse_losses
        
        return losses.mean()
    
    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prompt_len: int
    ) -> torch.Tensor:
        """
        计算 batch 的平均 log 概率
        
        只计算 response 部分（不包括 prompt）的 log 概率。
        
        参数：
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            prompt_len: prompt 的长度
        
        返回：
            [batch_size] 每个样本的平均 log 概率
        """
        # 只计算 response 部分
        # logits: [batch, seq-1, vocab], labels: [batch, seq-1]
        logits = logits[:, prompt_len:-1, :]
        labels = labels[:, prompt_len+1:]
        
        # 计算每个 token 的 log 概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 收集每个 token 的 log 概率
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        # 计算平均 log 概率（对每个序列）
        return per_token_logps.mean(dim=1)
    
    def _count_tokens(self, batch: Dict[str, Any]) -> int:
        """计算 token 数量（chosen + rejected）"""
        chosen_tokens = batch.get('full_chosen_input_ids', batch.get('chosen_input_ids'))
        rejected_tokens = batch.get('full_rejected_input_ids', batch.get('rejected_input_ids'))
        
        total = 0
        if chosen_tokens is not None:
            total += chosen_tokens.numel()
        if rejected_tokens is not None:
            total += rejected_tokens.numel()
        
        return total
    
    def _get_extra_config_info(self) -> Dict[str, Any]:
        """提供 DPO 特定的配置信息"""
        return {
            '训练类型': 'DPO (Direct Preference Optimization)',
            '损失函数': 'DPO Contrastive Loss',
            'Beta (温度)': self.beta,
            '标签平滑': self.label_smoothing,
            '参考模型': '已加载（冻结）'
        }
    
    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        """保存 DPO 特定的状态"""
        return {
            'beta': self.beta,
            'label_smoothing': self.label_smoothing
        }
    
    def _load_extra_checkpoint_state(self, extra_state: Dict[str, Any]):
        """加载 DPO 特定的状态"""
        self.beta = extra_state.get('beta', self.beta)
        self.label_smoothing = extra_state.get('label_smoothing', self.label_smoothing)
