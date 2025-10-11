# Trainer 示例代码

本目录包含训练器的示例代码和参考实现。

## 📁 文件说明

### `basic_trainer_example.py`
这是最初的训练器实现，展示了一个完整的训练器应该包含的所有组件：
- 分布式训练支持
- 混合精度训练
- 梯度累积和裁剪
- 检查点管理
- 完整的训练循环

**用途**：作为学习参考，了解训练器的完整实现细节。

**注意**：实际项目中应使用新的模块化训练器（`SFTTrainer`、`DPOTrainer` 等）。

---

### `distributed_train_example.py`
完整的分布式训练示例，展示了如何使用训练器进行单机多卡或多机多卡训练。

**包含内容**：
- 分布式环境初始化
- `DistributedSampler` 的使用
- 数据加载器配置
- 模型、优化器、调度器的创建
- 完整的训练流程

**运行方式**：

```bash
# 单机单卡（普通训练）
python distributed_train_example.py

# 单机4卡
torchrun --nproc_per_node=4 distributed_train_example.py

# 多机多卡（2机8卡，每机4卡）
# 主节点（节点0）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.1" --master_port=29500 \
         distributed_train_example.py

# 从节点（节点1）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="192.168.1.1" --master_port=29500 \
         distributed_train_example.py
```

---

## 🚀 如何使用新的训练器

### 1. SFT 训练（监督微调）

```python
from src.trainer import SFTTrainer
from src.trainer.optimizer import configure_optimizer

# 配置优化器
optimizer = configure_optimizer(
    model=model,
    optimizer_type='adamw',
    lr=1e-4,
    weight_decay=0.01
)

# 创建训练器
trainer = SFTTrainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    max_epochs=10,
    grad_accum_steps=4,
    use_amp=True
)

# 开始训练
trainer.train()
```

### 2. DPO 训练（偏好对齐）

```python
import copy
from src.trainer import DPOTrainer

# 创建参考模型（SFT 后的模型副本）
ref_model = copy.deepcopy(model)
ref_model.eval()

# 创建 DPO 训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    optimizer=optimizer,
    train_loader=dpo_train_loader,  # DPO 格式数据
    val_loader=dpo_val_loader,
    beta=0.1,  # DPO 温度参数
    device='cuda',
    max_epochs=3
)

# 开始训练
trainer.train()
```

### 3. GRPO 训练（强化学习）

```python
from src.trainer import GRPOTrainer

# 注意：GRPO 需要自定义采样和奖励逻辑
# 这里展示基本用法，实际使用需要提供 responses 和 rewards

trainer = GRPOTrainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,  # 需要包含 responses 和 rewards
    num_samples_per_prompt=4,
    clip_eps=0.2,
    device='cuda',
    max_epochs=5
)

# 开始训练
trainer.train()
```

### 4. 分布式训练（自动检测）

所有训练器都支持分布式训练，无需额外配置：

```bash
# 使用 torchrun 启动
torchrun --nproc_per_node=4 train_sft.py

# 训练器会自动检测并启用 DDP
```

---

## 📚 更多资源

- **教学版本**：查看 `../tutorial/` 目录了解从零实现的训练器
- **实际训练**：查看父目录的模块化训练器实现
- **文档**：查看 `../README.md` 了解完整的模块文档

---

## ⚠️ 注意事项

1. **示例代码仅供参考**：实际项目请使用模块化的训练器（`SFTTrainer` 等）
2. **数据格式**：确保数据加载器返回的 batch 格式与训练器要求一致
3. **分布式训练**：使用 `DistributedSampler` 确保数据不重复
4. **检查点**：只有主进程保存检查点，所有进程加载检查点

---

**更新时间**：2025-10-11  
**作者**：AnotherLLMFromScratch 项目

