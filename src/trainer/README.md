# Trainer 模块

本模块提供训练器和优化器的实现，包含**教学版本**和**实际训练版本**两种实现方式。

## 📂 目录结构

```
trainer/
├── tutorial/                          # 教学版本（手搓实现，详细注释）
│   ├── __init__.py                   # 模块初始化
│   ├── optimizer_from_scratch.py     # 从零实现优化器
│   └── trainer_from_scratch.py       # 从零实现训练器
├── optimizer.py                       # 实际训练版本（精简高效）
├── trainer.py                         # 实际训练版本（精简高效）
├── checkpoint.py                      # 检查点管理工具
└── README.md                          # 本文档
```

## 🎯 两种版本的区别

### 教学版本（`tutorial/`）

**目的**：帮助理解深度学习训练的底层原理

**特点**：
- ✍️ 从零手搓实现，不依赖 PyTorch 内置优化器
- 📚 包含详细的数学公式和实现原理注释
- 🔍 逐步解释每个步骤的作用
- 🎓 适合学习和教学

**包含内容**：
- `AdamWFromScratch`: 手搓 AdamW 优化器（带公式推导）
- `SGDFromScratch`: 手搓 SGD 优化器（带动量）
- `TrainerFromScratch`: 手搓训练循环（详细步骤注释）

**示例**：
```python
from src.trainer.tutorial import AdamWFromScratch, TrainerFromScratch

# 使用手搓的优化器
optimizer = AdamWFromScratch(
    params=model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# 使用手搓的训练器
trainer = TrainerFromScratch(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader
)

trainer.train()
```

### 实际训练版本（根目录）

**目的**：提供高效、可靠的训练实现

**特点**：
- ⚡ 使用 PyTorch 内置优化器和工具
- 🚀 代码精简，性能优化
- 🎯 适合实际项目使用
- 🛠️ 提供便捷的工厂函数和配置选项

**包含内容**：
- `create_optimizer()`: 优化器工厂函数
- `configure_optimizer()`: 带参数分组的优化器配置
- `Trainer`: 高效的训练器类

**示例**：
```python
from src.trainer.optimizer import configure_optimizer
from src.trainer.trainer import Trainer

# 使用工厂函数创建优化器（自动参数分组）
optimizer = configure_optimizer(
    model=model,
    optimizer_type='adamw',
    lr=1e-3,
    weight_decay=0.01
)

# 使用精简的训练器
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=1000)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=scheduler,
    device='cuda',
    max_epochs=10,
    grad_accum_steps=4,
    use_amp=True
)

trainer.train()
```

## 📖 推荐学习路径

### 第一步：理解优化器原理
阅读 `tutorial/optimizer_from_scratch.py`

**学习要点**：
1. Adam/AdamW 的数学原理
2. 一阶矩和二阶矩的计算
3. 偏差修正的作用
4. 权重衰减的实现方式

### 第二步：理解训练循环
阅读 `tutorial/trainer_from_scratch.py`

**学习要点**：
1. 前向传播和反向传播
2. 梯度累积的原理
3. 梯度裁剪防止爆炸
4. 混合精度训练
5. 学习率调度

### 第三步：学习工程实践
阅读 `optimizer.py` 和 `trainer.py`

**学习要点**：
1. 如何使用 PyTorch 内置工具
2. 参数分组的最佳实践
3. 高效的训练循环实现
4. 检查点管理

## 🚀 快速开始

### 使用教学版本（学习用）

```python
from src.trainer.tutorial import AdamWFromScratch, TrainerFromScratch

# 创建优化器
optimizer = AdamWFromScratch(
    params=model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# 创建训练器
trainer = TrainerFromScratch(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    max_epochs=10
)

# 开始训练
trainer.train()
```

### 使用实际版本（实际项目）

```python
from src.trainer.optimizer import configure_optimizer
from src.trainer.trainer import Trainer
from torch.optim.lr_scheduler import CosineAnnealingLR

# 配置优化器（自动参数分组）
optimizer = configure_optimizer(
    model=model,
    optimizer_type='adamw',
    lr=1e-3,
    weight_decay=0.01
)

# 创建学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

# 创建训练器
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    scheduler=scheduler,
    device='cuda',
    max_epochs=10,
    grad_accum_steps=4,
    max_grad_norm=1.0,
    use_amp=True,
    log_interval=100,
    save_dir='./checkpoints'
)

# 开始训练
trainer.train()
```

## 💡 核心概念解释

### 1. 梯度累积（Gradient Accumulation）

**作用**：模拟更大的 batch size，在显存有限时很有用

**原理**：
```python
# 累积 N 步的梯度
loss = loss / N
loss.backward()  # 累积梯度

# 每 N 步更新一次参数
if step % N == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 2. 梯度裁剪（Gradient Clipping）

**作用**：防止梯度爆炸，稳定训练

**原理**：
```python
# 如果梯度范数 > max_norm，缩放梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 3. 混合精度训练（Mixed Precision）

**作用**：加速训练，节省显存

**原理**：
```python
with autocast('cuda'):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. 参数分组（Parameter Groups）

**作用**：对不同参数应用不同的优化策略

**最佳实践**：
- bias 和 LayerNorm 参数通常不应用权重衰减
- 权重矩阵应用权重衰减

```python
param_groups = [
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
]
optimizer = AdamW(param_groups, lr=1e-3)
```

### 5. 学习率调度（Learning Rate Scheduling）

**常用策略**：
- **Warmup**：训练初期线性增长学习率
- **Cosine Annealing**：余弦函数衰减
- **Step Decay**：每隔一定步数降低学习率

```python
# PyTorch 提供的调度器
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,      # 余弦退火
    CosineAnnealingWarmRestarts,  # 带重启的余弦退火
    OneCycleLR,             # One Cycle 策略
    ReduceLROnPlateau       # 基于验证集的自适应调整
)
```

## 📊 检查点管理

训练器自动保存检查点，包含：
- 模型权重
- 优化器状态
- 调度器状态
- 训练进度（epoch, step）
- 最佳验证损失

```python
# 保存检查点
trainer.save_checkpoint(is_best=True)

# 加载检查点
trainer.load_checkpoint('./checkpoints/best_model.pt')

# 继续训练
trainer.train()
```

## 🎓 教学资源

### 推荐阅读顺序

1. **优化器基础**
   - `tutorial/optimizer_from_scratch.py` 中的 SGD 实现
   - 理解梯度下降和动量

2. **AdamW 优化器**
   - `tutorial/optimizer_from_scratch.py` 中的 AdamW 实现
   - 理解自适应学习率和权重衰减

3. **训练循环**
   - `tutorial/trainer_from_scratch.py` 的完整实现
   - 理解训练的完整流程

4. **工程实践**
   - `optimizer.py` 的工厂模式和参数分组
   - `trainer.py` 的高效实现

### 实验建议

1. **对比实验**：使用相同超参数，对比手搓版本和 PyTorch 版本的结果
2. **参数调优**：尝试不同的学习率、权重衰减、梯度裁剪阈值
3. **性能分析**：测量混合精度训练的加速比和显存节省
4. **可视化**：使用 tensorboard 或 wandb 记录训练曲线

## 📝 注意事项

1. **教学版本仅用于学习**，实际项目请使用精简版本
2. **参数分组**是训练 Transformer 的最佳实践
3. **混合精度**在 Ampere 架构（RTX 30 系列）及以上有显著加速
4. **梯度裁剪**对于 RNN/LSTM/Transformer 训练很重要
5. **学习率预热**可以提高训练稳定性

## 🔗 相关资源

- [Adam 论文](https://arxiv.org/abs/1412.6980)
- [AdamW 论文](https://arxiv.org/abs/1711.05101)
- [混合精度训练](https://pytorch.org/docs/stable/amp.html)
- [梯度裁剪](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

---

**作者**：AnotherLLMFromScratch 项目  
**更新时间**：2025-10-11
