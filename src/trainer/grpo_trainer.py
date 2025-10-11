"""
GRPO 训练器 - Group Relative Policy Optimization Trainer
==========================================================

用于组相对策略优化（Group Relative Policy Optimization）的训练器。

GRPO 原理：
- 基于 PPO（Proximal Policy Optimization）的改进
- 使用组内对比（group-wise comparison）代替价值函数
- 不需要训练 critic 网络，更高效
- 通过奖励优势（advantage）来优化策略

注意：GRPO 需要在线采样和奖励计算，实现较为复杂。
本实现提供框架，实际使用时需要根据具体任务调整采样和奖励逻辑。

作者：AnotherLLMFromScratch 项目
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """
    GRPO（组相对策略优化）训练器
    
    使用强化学习的方式训练模型，通过在线采样和奖励信号来优化策略。
    
    参数：
        model: 策略模型（要训练的模型）
        optimizer: 优化器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_samples_per_prompt: 每个 prompt 采样的回答数量（默认 4）
        clip_eps: PPO 裁剪参数（默认 0.2）
        kl_coef: KL 散度惩罚系数（默认 0.1）
        entropy_coef: 熵奖励系数（默认 0.01，鼓励探索）
        temperature: 采样温度（默认 1.0）
        use_baseline: 是否使用组内平均作为 baseline（默认 True）
        **kwargs: 其他 BaseTrainer 参数
    
    注意：
        GRPO 实现较为复杂，需要：
        1. 在线采样生成回答
        2. 计算奖励（需要奖励模型或奖励函数）
        3. 计算优势函数和 PPO 损失
        
        本实现提供基础框架，实际使用需要根据任务定制。
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        train_loader,
        val_loader=None,
        num_samples_per_prompt: int = 4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.1,
        entropy_coef: float = 0.01,
        temperature: float = 1.0,
        use_baseline: bool = True,
        **kwargs
    ):
        # 保存 GRPO 特定参数
        self.num_samples = num_samples_per_prompt
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.temperature = temperature
        self.use_baseline = use_baseline
        
        # 初始化基类
        super().__init__(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            **kwargs
        )
        
        # 警告：GRPO 需要额外实现
        if self.is_main:
            print("\n⚠️  警告：GRPO 训练器需要自定义采样和奖励逻辑！")
            print("请根据具体任务实现以下方法：")
            print("  - _sample_responses(): 在线采样回答")
            print("  - _compute_rewards(): 计算奖励分数")
            print("  - _compute_old_logprobs(): 计算旧策略的 log 概率\n")
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备 GRPO 训练的 batch
        
        GRPO 通常需要在线采样，如果 batch 中没有预先生成的 responses 和 rewards，
        则需要在这里进行采样（实际实现需要定制）。
        
        参数：
            batch: 原始 batch
        
        返回：
            处理后的 batch 字典
        """
        prepared = {}
        
        # 移动所有张量到设备
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value
        
        # 如果没有预先生成的数据，需要在线采样（需要实现）
        if 'responses' not in prepared or 'rewards' not in prepared:
            raise NotImplementedError(
                "GRPO 需要在线采样，请实现 _sample_responses() 方法或在数据集中提供 responses 和 rewards"
            )
        
        return prepared
    
    def _forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        GRPO 的前向传播
        
        计算当前策略的 log 概率。
        
        参数：
            batch: 准备好的 batch
        
        返回：
            包含 logprobs 的字典
        """
        prompts = batch.get('prompts', batch.get('input_ids'))
        responses = batch['responses']
        
        if prompts is None:
            raise ValueError("Batch must contain 'prompts' or 'input_ids'")
        
        # 拼接 prompt + response
        full_seqs = torch.cat([prompts, responses], dim=1)
        
        # 前向传播
        outputs = self.model(full_seqs)
        
        # 处理模型输出
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            raise ValueError(f"Unsupported model output format: {type(outputs)}")
        
        # 计算 log 概率（简化实现）
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 这里需要更复杂的逻辑来计算 response 部分的 log 概率
        # 简化版本：返回平均 log 概率
        prompt_len = prompts.size(1)
        response_len = responses.size(1)
        
        # 收集生成部分的 log 概率
        token_logprobs = []
        for i in range(response_len - 1):
            pos = prompt_len + i
            if pos < log_probs.size(1):
                token_ids = responses[:, i + 1]
                token_logprob = log_probs[:, pos, :].gather(1, token_ids.unsqueeze(1)).squeeze(1)
                token_logprobs.append(token_logprob)
        
        if token_logprobs:
            avg_logprobs = torch.stack(token_logprobs, dim=1).mean(dim=1)
        else:
            avg_logprobs = torch.zeros(prompts.size(0), device=self.device)
        
        return {
            'logprobs': avg_logprobs,
            'logits': logits
        }
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算 GRPO 损失
        
        GRPO 损失 = PPO 损失 + KL 惩罚 + 熵奖励
        
        参数：
            outputs: 包含 logprobs 的字典
            batch: 包含 rewards 和 old_logprobs 的 batch
        
        返回：
            标量损失值
        """
        new_logprobs = outputs['logprobs']
        old_logprobs = batch.get('old_logprobs', new_logprobs.detach())
        rewards = batch.get('rewards', torch.zeros_like(new_logprobs))
        
        # 计算优势函数（使用组内平均作为 baseline）
        if self.use_baseline and len(rewards) >= self.num_samples:
            # 重塑为 [batch_size // num_samples, num_samples]
            batch_size = rewards.size(0)
            num_groups = batch_size // self.num_samples
            
            if num_groups > 0:
                rewards_grouped = rewards[:num_groups * self.num_samples].view(num_groups, self.num_samples)
                
                # 计算组内平均
                baseline = rewards_grouped.mean(dim=1, keepdim=True)
                baseline = baseline.expand(-1, self.num_samples).reshape(-1)
                
                # 填充剩余的样本
                if batch_size > num_groups * self.num_samples:
                    baseline = torch.cat([baseline, rewards[num_groups * self.num_samples:]])
                
                advantages = rewards - baseline
            else:
                advantages = rewards
        else:
            advantages = rewards
        
        # 标准化优势（可选，提高稳定性）
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算重要性权重
        # ratio = exp(new_logprobs - old_logprobs) = π_new / π_old
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # PPO 裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 散度惩罚（防止策略变化太大）
        kl_penalty = (new_logprobs - old_logprobs).mean()
        kl_loss = self.kl_coef * torch.abs(kl_penalty)
        
        # 熵奖励（鼓励探索）
        entropy = -(new_logprobs.exp() * new_logprobs).mean()
        entropy_bonus = -self.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + kl_loss + entropy_bonus
        
        return total_loss
    
    def _count_tokens(self, batch: Dict[str, Any]) -> int:
        """计算 token 数量"""
        if 'responses' in batch:
            return batch['responses'].numel()
        elif 'input_ids' in batch:
            return batch['input_ids'].numel()
        return 0
    
    def _get_extra_config_info(self) -> Dict[str, Any]:
        """提供 GRPO 特定的配置信息"""
        return {
            '训练类型': 'GRPO (Group Relative Policy Optimization)',
            '损失函数': 'PPO + KL Penalty + Entropy',
            '每 prompt 采样数': self.num_samples,
            'Clip Epsilon': self.clip_eps,
            'KL 系数': self.kl_coef,
            '熵系数': self.entropy_coef,
            '温度': self.temperature,
            '使用 Baseline': self.use_baseline
        }
    
    def _get_extra_log_info(self, batch: Dict[str, Any]) -> str:
        """GRPO 额外日志信息"""
        if 'rewards' in batch:
            avg_reward = batch['rewards'].mean().item()
            return f"Avg Reward: {avg_reward:.4f}"
        return ""
