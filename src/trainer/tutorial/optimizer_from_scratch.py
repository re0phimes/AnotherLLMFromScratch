"""
手搓优化器 - 教学版本
========================

本模块从零实现 AdamW 优化器，用于教学目的。
包含详细的数学原理和实现细节注释。

AdamW 优化器原理：
- Adam (Adaptive Moment Estimation): 自适应矩估计
- W (Weight Decay): 权重衰减（L2 正则化的改进版本）

核心公式：
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t                 # 一阶矩（动量）
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²                # 二阶矩（方差）
    m̂_t = m_t / (1 - β₁^t)                              # 偏差修正
    v̂_t = v_t / (1 - β₂^t)                              # 偏差修正
    θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})  # 参数更新

作者：AnotherLLMFromScratch 项目
用途：教学演示
"""

import math
from typing import List, Dict, Optional, Tuple
import torch
from torch import Tensor


class AdamWFromScratch:
    """
    从零实现的 AdamW 优化器
    
    AdamW 是 Adam 优化器的改进版本，主要区别在于权重衰减的实现方式：
    - Adam: 权重衰减添加到梯度中（L2 正则化）
    - AdamW: 权重衰减直接应用于参数更新（解耦权重衰减）
    
    参数说明：
        params: 需要优化的参数列表
        lr: 学习率 (learning rate)，默认 1e-3
        betas: (β₁, β₂) 用于计算梯度及其平方的移动平均，默认 (0.9, 0.999)
        eps: 数值稳定性常数，防止除零，默认 1e-8
        weight_decay: 权重衰减系数（L2 惩罚），默认 0.01
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        # 验证参数有效性
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta₁: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta₂: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"无效的 epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay: {weight_decay}")
        
        # 保存超参数
        self.params = list(params)  # 需要优化的参数列表
        self.lr = lr                # 学习率
        self.beta1, self.beta2 = betas  # β₁ 和 β₂
        self.eps = eps              # 数值稳定性常数
        self.weight_decay = weight_decay  # 权重衰减系数
        
        # 初始化优化器状态
        # state 字典保存每个参数的优化状态：
        # - step: 当前迭代步数
        # - exp_avg: 梯度的指数移动平均（一阶矩 m_t）
        # - exp_avg_sq: 梯度平方的指数移动平均（二阶矩 v_t）
        self.state: Dict[int, Dict[str, Tensor]] = {}
        
        # 为每个参数初始化状态
        for i, param in enumerate(self.params):
            self.state[i] = {
                'step': 0,  # 初始步数为 0
                'exp_avg': torch.zeros_like(param),     # m_0 = 0
                'exp_avg_sq': torch.zeros_like(param),  # v_0 = 0
            }
    
    def zero_grad(self):
        """
        清零所有参数的梯度
        
        在每次反向传播之前需要调用此方法，因为 PyTorch 会累积梯度。
        """
        for param in self.params:
            if param.grad is not None:
                # 将梯度张量清零
                param.grad.zero_()
    
    @torch.no_grad()  # 禁用梯度计算，优化器更新不需要梯度
    def step(self):
        """
        执行一步参数更新
        
        实现 AdamW 优化器的完整更新逻辑：
        1. 获取当前梯度 g_t
        2. 更新一阶矩和二阶矩的移动平均
        3. 计算偏差修正后的估计
        4. 应用权重衰减和自适应学习率更新参数
        """
        
        # 遍历所有需要优化的参数
        for i, param in enumerate(self.params):
            # 跳过没有梯度的参数（例如冻结的参数）
            if param.grad is None:
                continue
            
            # ==================== 步骤 1: 获取梯度 ====================
            grad = param.grad  # g_t: 当前参数的梯度
            
            # 获取该参数的优化状态
            state = self.state[i]
            exp_avg = state['exp_avg']        # m_{t-1}: 一阶矩
            exp_avg_sq = state['exp_avg_sq']  # v_{t-1}: 二阶矩
            
            # 更新步数
            state['step'] += 1
            step = state['step']
            
            # ==================== 步骤 2: 更新移动平均 ====================
            # 一阶矩更新: m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            # 这是梯度的指数移动平均，类似于动量
            exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            
            # 二阶矩更新: v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            # 这是梯度平方的指数移动平均，用于自适应学习率
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            
            # ==================== 步骤 3: 偏差修正 ====================
            # 由于 m_0 = v_0 = 0，在训练初期 m_t 和 v_t 会偏向于 0
            # 偏差修正因子补偿这种偏差
            
            # 计算偏差修正因子
            bias_correction1 = 1 - self.beta1 ** step  # 1 - β₁^t
            bias_correction2 = 1 - self.beta2 ** step  # 1 - β₂^t
            
            # 偏差修正后的一阶矩: m̂_t = m_t / (1 - β₁^t)
            corrected_exp_avg = exp_avg / bias_correction1
            
            # 偏差修正后的二阶矩: v̂_t = v_t / (1 - β₂^t)
            corrected_exp_avg_sq = exp_avg_sq / bias_correction2
            
            # ==================== 步骤 4: 计算自适应学习率 ====================
            # 分母: √v̂_t + ε
            # sqrt() 取平方根，add_() 加上 epsilon 防止除零
            denom = corrected_exp_avg_sq.sqrt().add_(self.eps)
            
            # 自适应梯度: m̂_t / (√v̂_t + ε)
            # 这使得学习率能够根据梯度的历史信息自适应调整
            adaptive_grad = corrected_exp_avg / denom
            
            # ==================== 步骤 5: 应用权重衰减（AdamW 的关键特性）====================
            # AdamW 的创新：权重衰减与梯度更新解耦
            # 直接对参数应用衰减，而不是添加到梯度中
            if self.weight_decay != 0:
                # θ_t = θ_{t-1} * (1 - lr * λ)
                # 这等价于 θ_t = θ_{t-1} - lr * λ * θ_{t-1}
                param.mul_(1 - self.lr * self.weight_decay)
            
            # ==================== 步骤 6: 参数更新 ====================
            # θ_t = θ_t - lr * adaptive_grad
            # 注意：如果应用了权重衰减，这里的 θ_t 已经是衰减后的值
            param.add_(adaptive_grad, alpha=-self.lr)
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.lr
    
    def set_lr(self, lr: float):
        """设置新的学习率"""
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        self.lr = lr
    
    def state_dict(self) -> Dict:
        """
        保存优化器状态
        
        返回包含完整优化器状态的字典，用于检查点保存。
        """
        return {
            'state': self.state,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        加载优化器状态
        
        从保存的字典恢复优化器状态，用于检查点恢复。
        """
        self.state = state_dict['state']
        self.lr = state_dict['lr']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.eps = state_dict['eps']
        self.weight_decay = state_dict['weight_decay']


class SGDFromScratch:
    """
    从零实现的 SGD (随机梯度下降) 优化器
    
    SGD 是最基础的优化算法，更新公式：
        θ_t = θ_{t-1} - lr * g_t
    
    带动量的 SGD (Momentum):
        v_t = μ * v_{t-1} + g_t
        θ_t = θ_{t-1} - lr * v_t
    
    参数说明：
        params: 需要优化的参数列表
        lr: 学习率
        momentum: 动量系数，默认 0（不使用动量）
        weight_decay: 权重衰减系数，默认 0
    """
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if momentum < 0.0:
            raise ValueError(f"无效的动量: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay: {weight_decay}")
        
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # 初始化动量缓冲区
        self.velocity: Dict[int, Tensor] = {}
        if self.momentum > 0:
            for i, param in enumerate(self.params):
                self.velocity[i] = torch.zeros_like(param)
    
    def zero_grad(self):
        """清零梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    @torch.no_grad()
    def step(self):
        """执行一步参数更新"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # 应用权重衰减（L2 正则化）
            if self.weight_decay != 0:
                grad = grad.add(param, alpha=self.weight_decay)
            
            # 应用动量
            if self.momentum > 0:
                v = self.velocity[i]
                # v_t = μ * v_{t-1} + g_t
                v.mul_(self.momentum).add_(grad)
                # θ_t = θ_{t-1} - lr * v_t
                param.add_(v, alpha=-self.lr)
            else:
                # 标准 SGD: θ_t = θ_{t-1} - lr * g_t
                param.add_(grad, alpha=-self.lr)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    """
    演示如何使用手搓的优化器
    """
    print("=" * 60)
    print("手搓优化器教学演示")
    print("=" * 60)
    
    # 创建一个简单的模型参数
    param = torch.randn(10, 10, requires_grad=True)
    
    # 初始化 AdamW 优化器
    optimizer = AdamWFromScratch(
        params=[param],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    print(f"初始参数范数: {param.norm().item():.6f}")
    
    # 模拟训练循环
    for step in range(5):
        # 模拟一个简单的损失函数: loss = sum(param²)
        loss = (param ** 2).sum()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 反向传播
        loss.backward()
        
        print(f"\n步骤 {step + 1}:")
        print(f"  损失: {loss.item():.6f}")
        print(f"  梯度范数: {param.grad.norm().item():.6f}")
        
        # 更新参数
        optimizer.step()
        
        print(f"  更新后参数范数: {param.norm().item():.6f}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)

