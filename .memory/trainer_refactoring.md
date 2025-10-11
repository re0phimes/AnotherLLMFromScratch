# Trainer 模块重构完成记录

**日期**: 2025-10-11  
**任务**: 重构 Trainer 模块，支持多种训练范式（SFT、DPO、GRPO）

---

## ✅ 完成的工作

### 1. 创建模块化训练器架构

#### 1.1 基础训练器 (`base_trainer.py`)
- ✅ 抽象基类，包含所有通用训练逻辑
- ✅ 分布式训练支持（自动检测 DDP）
- ✅ 混合精度训练（AMP）
- ✅ 梯度累积和梯度裁剪
- ✅ 学习率调度
- ✅ 检查点管理（保存/加载）
- ✅ 定义抽象方法供子类实现：
  - `_prepare_batch()`: 准备批次数据
  - `_forward()`: 前向传播
  - `_compute_loss()`: 计算损失

**核心设计**：
- 所有通用逻辑在基类中实现
- 任务特定逻辑由子类实现
- 最大化代码复用

#### 1.2 SFT 训练器 (`sft_trainer.py`)
- ✅ 继承 `BaseTrainer`
- ✅ 实现标准交叉熵损失
- ✅ 支持指令微调和预训练
- ✅ 自动处理 prompt 部分的 label (-100)
- ✅ 提供 `PretrainTrainer` 别名

**数据格式**：
```python
{
    'input_ids': [B, L],
    'labels': [B, L],  # prompt 部分为 -100
    'attention_mask': [B, L]  # 可选
}
```

#### 1.3 DPO 训练器 (`dpo_trainer.py`)
- ✅ 继承 `BaseTrainer`
- ✅ 实现 DPO 对比损失
- ✅ 维护冻结的参考模型（reference model）
- ✅ 对 chosen 和 rejected 分别前向传播
- ✅ 支持标签平滑
- ✅ 可配置 beta 温度参数

**数据格式**：
```python
{
    'prompt_input_ids': [B, L],
    'chosen_input_ids': [B, L],
    'rejected_input_ids': [B, L],
    'prompt_attention_mask': [B, L],
    'chosen_attention_mask': [B, L],
    'rejected_attention_mask': [B, L]
}
```

**核心逻辑**：
- Policy model 和 reference model 都前向传播
- 计算 log 概率差值
- DPO 损失：`-log(σ(β * (π_θ - π_ref)))`

#### 1.4 GRPO 训练器 (`grpo_trainer.py`)
- ✅ 继承 `BaseTrainer`
- ✅ 实现 PPO 风格的策略优化
- ✅ 支持在线采样（生成回答）
- ✅ 组内对比计算优势函数
- ✅ 支持奖励模型或自定义奖励函数
- ✅ 包含 KL 惩罚和熵奖励

**数据格式**：
```python
{
    'prompts': [B, L],  # 输入
    # 以下可以在线生成：
    'responses': [B, L],
    'rewards': [B],
    'old_logprobs': [B]
}
```

**核心逻辑**：
- 每个 prompt 采样多个回答
- 计算奖励和优势函数
- PPO 裁剪损失 + KL 惩罚 + 熵奖励

---

### 2. 模块组织结构

```
src/trainer/
├── base_trainer.py          # 基类（600+ 行）
├── sft_trainer.py           # SFT 训练器（300+ 行）
├── dpo_trainer.py           # DPO 训练器（500+ 行）
├── grpo_trainer.py          # GRPO 训练器（600+ 行）
├── __init__.py              # 模块初始化和导出
├── optimizer.py             # 优化器工具（已有）
├── checkpoint.py            # 检查点工具（已有）
├── tutorial/                # 教学版本（已有）
│   ├── __init__.py
│   ├── optimizer_from_scratch.py
│   └── trainer_from_scratch.py
└── examples/                # 示例代码（新增）
    ├── README.md
    ├── basic_trainer_example.py      # 原 trainer.py
    └── distributed_train_example.py   # 分布式示例
```

---

### 3. 模块接口设计

#### 3.1 统一的导入接口

```python
from src.trainer import (
    BaseTrainer,
    SFTTrainer,
    PretrainTrainer,
    DPOTrainer,
    GRPOTrainer,
    configure_optimizer,
    get_trainer_class,
    create_trainer
)
```

#### 3.2 工厂函数

```python
# 方式 1：直接创建
trainer = SFTTrainer(model, optimizer, train_loader)

# 方式 2：使用工厂函数
trainer = create_trainer(
    'sft',
    model=model,
    optimizer=optimizer,
    train_loader=train_loader
)

# 方式 3：动态选择
TrainerClass = get_trainer_class('dpo')
trainer = TrainerClass(...)
```

---

## 📊 训练器对比

| 特性 | SFT | DPO | GRPO |
|-----|-----|-----|------|
| 损失函数 | 交叉熵 | 对比损失 | PPO + KL |
| 数据类型 | 指令-回答 | 偏好对比 | 提示（在线生成）|
| 参考模型 | ❌ | ✅ 需要 | ❌ |
| 奖励模型 | ❌ | ❌ | ✅ 可选 |
| 在线采样 | ❌ | ❌ | ✅ 需要 |
| 训练速度 | 快 | 中 | 慢 |
| 显存占用 | 低 | 中（2x模型）| 中 |
| 适用场景 | 指令微调 | 偏好对齐 | RLHF |

---

## 🎯 使用场景

### SFT (Supervised Fine-Tuning)
**适用于**：
- 指令微调（Instruction Tuning）
- 预训练（Pretraining）
- 任务特定微调

**数据需求**：
- 指令-回答对
- 高质量的标注数据

### DPO (Direct Preference Optimization)
**适用于**：
- 偏好对齐（Preference Alignment）
- RLHF 的简化替代
- 模型行为调整

**数据需求**：
- 偏好对比数据（chosen vs rejected）
- 可以从多个模型输出中构建

### GRPO (Group Relative Policy Optimization)
**适用于**：
- 强化学习式对齐
- 需要探索的场景
- 复杂的奖励函数

**数据需求**：
- 提示（prompts）
- 奖励模型或奖励函数

---

## 🔄 训练流程示例

### 典型的三阶段训练

```
阶段 1: 预训练
└─> PretrainTrainer (大规模文本数据)

阶段 2: 指令微调
└─> SFTTrainer (指令-回答数据)

阶段 3: 偏好对齐（选择其一）
├─> DPOTrainer (离线偏好数据)
└─> GRPOTrainer (在线强化学习)
```

---

## 📝 关键实现细节

### 1. 抽象方法设计

基类定义三个抽象方法，子类必须实现：

```python
class BaseTrainer(ABC):
    @abstractmethod
    def _prepare_batch(self, batch):
        """准备数据，移到设备"""
        pass
    
    @abstractmethod
    def _forward(self, batch):
        """前向传播，返回输出"""
        pass
    
    @abstractmethod
    def _compute_loss(self, outputs, batch):
        """计算损失"""
        pass
```

### 2. 扩展点设计

基类提供可选的扩展点：

```python
class BaseTrainer:
    def _get_extra_config_info(self):
        """子类添加配置信息"""
        return {}
    
    def _get_extra_log_info(self, batch):
        """子类添加日志信息"""
        return ""
    
    def _get_extra_checkpoint_state(self):
        """子类保存额外状态"""
        return None
    
    def _count_tokens(self, batch):
        """子类自定义 token 计数"""
        return batch['input_ids'].numel()
```

### 3. 分布式训练处理

所有训练器自动支持分布式：

```python
# 自动检测环境
self.is_distributed = is_distributed()
self.rank = get_rank()

# 自动包装模型
if self.use_ddp:
    self.model = nn.parallel.DistributedDataParallel(...)

# 自动同步指标
if self.is_distributed:
    avg_loss = reduce_tensor(avg_loss_tensor, op='mean')

# 只在主进程打印和保存
if self.is_main:
    print(...)
    self.save_checkpoint(...)
```

---

## 🚀 性能优化

### 1. 混合精度训练
- 使用 `torch.amp.autocast` 和 `GradScaler`
- 自动在 CUDA 上启用
- 可节省约 50% 显存

### 2. 梯度累积
- 模拟更大的 batch size
- 在显存有限时很有用
- 梯度正确累积并归一化

### 3. 梯度裁剪
- 防止梯度爆炸
- 使用全局范数裁剪
- 默认阈值 1.0

---

## 📦 依赖关系

```
BaseTrainer (基类)
├── utils.distributed (分布式工具)
├── torch.amp (混合精度)
└── torch.nn.parallel (DDP)

SFTTrainer (继承 BaseTrainer)
└── torch.nn.functional (交叉熵)

DPOTrainer (继承 BaseTrainer)
├── 参考模型 (ref_model)
└── 对比损失计算

GRPOTrainer (继承 BaseTrainer)
├── 奖励模型 (可选)
├── 在线采样逻辑
└── PPO 损失计算
```

---

## ⚡ 下一步计划

### 短期（可选）
- [ ] 添加 PPO 训练器（完整版本，带 critic）
- [ ] 添加 ORPO 训练器（Odds Ratio Preference Optimization）
- [ ] 添加训练指标可视化（TensorBoard/WandB）
- [ ] 添加早停（Early Stopping）功能

### 中期（可选）
- [ ] 支持 FSDP（更大模型）
- [ ] 支持模型量化训练
- [ ] 支持 LoRA/QLoRA 微调
- [ ] 添加更多学习率调度策略

### 长期（可选）
- [ ] 支持多任务学习
- [ ] 支持课程学习（Curriculum Learning）
- [ ] 添加更多的 RLHF 变体

---

## 📚 参考资料

### 论文
- **SFT**: Standard supervised learning
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **GRPO**: Group-based PPO variant

### 实现参考
- PyTorch DDP: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Transformers Trainer: https://github.com/huggingface/transformers
- TRL Library: https://github.com/huggingface/trl

---

## 🎓 学习建议

### 对于初学者
1. 阅读 `tutorial/trainer_from_scratch.py` 理解训练循环
2. 查看 `examples/` 中的示例代码
3. 使用 `SFTTrainer` 进行简单的微调实验

### 对于进阶用户
1. 阅读 `base_trainer.py` 理解架构设计
2. 根据需要继承 `BaseTrainer` 实现自定义训练器
3. 使用 DPO 或 GRPO 进行高级对齐训练

### 对于研究者
1. 理解不同训练范式的数学原理
2. 对比不同训练器的性能和效果
3. 基于现有框架实现新的训练算法

---

**总结**：本次重构创建了一个模块化、可扩展的训练器架构，支持主流的 LLM 训练范式，同时保持代码的清晰性和可维护性。所有训练器共享通用逻辑，只需实现任务特定的方法，大大提高了代码复用率。

**最后更新**：2025-10-11  
**状态**：✅ 完成

