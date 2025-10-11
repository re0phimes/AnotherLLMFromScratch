# Trainer 模块更新记录

**日期**: 2025-10-11
**任务**: 重构 Trainer 模块，添加教学版本和分布式训练支持

## 完成的工作

### 1. 创建教学版本（Tutorial）

#### 1.1 手搓优化器 (`src/trainer/tutorial/optimizer_from_scratch.py`)
- ✅ 从零实现 AdamW 优化器
- ✅ 详细的数学公式注释
- ✅ 逐步解释一阶矩、二阶矩、偏差修正
- ✅ 解释权重衰减的解耦实现
- ✅ 从零实现 SGD 优化器（带动量）
- ✅ 包含完整的使用示例

**核心概念**:
- Adam/AdamW 的数学原理
- 自适应学习率
- 动量和方差估计
- 偏差修正
- 解耦权重衰减

#### 1.2 手搓训练器 (`src/trainer/tutorial/trainer_from_scratch.py`)
- ✅ 从零实现完整训练循环
- ✅ 详细注释每个步骤
- ✅ 实现梯度累积
- ✅ 实现梯度裁剪
- ✅ 实现混合精度训练
- ✅ 实现学习率调度（预热+余弦退火）
- ✅ 实现检查点保存和加载
- ✅ 包含完整的使用示例

**核心概念**:
- 前向传播和反向传播
- 梯度累积原理
- 梯度裁剪防止爆炸
- 混合精度训练原理
- 学习率调度策略
- 训练/验证循环

#### 1.3 Tutorial 模块初始化 (`src/trainer/tutorial/__init__.py`)
- ✅ 导出所有教学版本的类
- ✅ 添加学习提示和使用指南
- ✅ 推荐学习路径

### 2. 精简实际训练版本

#### 2.1 优化器模块 (`src/trainer/optimizer.py`)
- ✅ 使用 PyTorch 内置优化器（AdamW, Adam, SGD）
- ✅ 提供 `create_optimizer()` 工厂函数
- ✅ 提供 `get_parameter_groups()` 参数分组函数
- ✅ 提供 `configure_optimizer()` 便捷配置函数
- ✅ 自动处理 bias 和 normalization 层的权重衰减

**最佳实践**:
- bias 参数不应用权重衰减
- LayerNorm/BatchNorm 参数不应用权重衰减
- 权重矩阵应用权重衰减

#### 2.2 训练器模块 (`src/trainer/trainer.py`)
- ✅ 使用 PyTorch 内置功能
- ✅ 支持混合精度训练（AMP）
- ✅ 支持梯度累积
- ✅ 支持梯度裁剪
- ✅ 支持学习率调度器
- ✅ 支持检查点管理
- ✅ **新增：完整的分布式训练支持（DDP）**

### 3. 集成分布式训练功能

#### 3.1 集成已有的 distributed.py 工具
- ✅ 导入分布式工具函数
- ✅ 自动检测分布式环境
- ✅ 自动初始化 rank, local_rank, world_size
- ✅ 自动选择设备（cuda:local_rank 或 cpu）

#### 3.2 DDP 模型包装
- ✅ 自动使用 DistributedDataParallel 包装模型
- ✅ 正确处理 device_ids 和 output_device
- ✅ 支持 CPU 和 GPU 分布式训练

#### 3.3 分布式训练的关键改进
- ✅ 训练损失跨进程同步（reduce_tensor）
- ✅ 验证损失跨进程同步
- ✅ Token 统计跨进程聚合
- ✅ 只在主进程打印日志
- ✅ 只在主进程保存检查点
- ✅ 正确保存/加载 DDP 模型（model.module）

#### 3.4 分布式训练示例 (`src/trainer/distributed_train_example.py`)
- ✅ 完整的分布式训练示例
- ✅ 演示 DistributedSampler 的使用
- ✅ 演示单机多卡训练
- ✅ 演示多机多卡训练
- ✅ 包含详细的运行命令

### 4. 文档更新

#### 4.1 README 更新 (`src/trainer/README.md`)
- ✅ 说明教学版本和实际版本的区别
- ✅ 提供推荐学习路径
- ✅ 添加核心概念解释
- ✅ 添加使用示例
- ✅ 添加分布式训练说明

## 使用方式

### 学习模式（使用教学版本）

```python
from src.trainer.tutorial import AdamWFromScratch, TrainerFromScratch

# 手搓优化器
optimizer = AdamWFromScratch(
    params=model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# 手搓训练器
trainer = TrainerFromScratch(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader
)

trainer.train()
```

### 实际训练模式（精简版本）

#### 单机训练
```python
from src.trainer.optimizer import configure_optimizer
from src.trainer.trainer import Trainer

# 配置优化器（自动参数分组）
optimizer = configure_optimizer(
    model=model,
    optimizer_type='adamw',
    lr=1e-3,
    weight_decay=0.01
)

# 创建训练器（自动检测单机/分布式）
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    max_epochs=10,
    use_amp=True
)

trainer.train()
```

#### 分布式训练
```bash
# 单机4卡
torchrun --nproc_per_node=4 train.py

# 多机多卡（2机8卡）
# 主节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.1" --master_port=29500 train.py

# 从节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="192.168.1.1" --master_port=29500 train.py
```

## 技术要点

### 分布式训练关键点

1. **环境检测**: 自动检测 `torchrun` 设置的环境变量
2. **模型包装**: 使用 `DistributedDataParallel` 包装模型
3. **数据采样**: 使用 `DistributedSampler` 确保数据不重复
4. **梯度同步**: DDP 自动同步梯度
5. **指标聚合**: 手动同步损失和统计指标
6. **检查点**: 只在主进程保存，所有进程加载
7. **日志输出**: 只在主进程打印

### 参数分组最佳实践

- ✅ 权重矩阵：应用权重衰减
- ❌ bias 参数：不应用权重衰减
- ❌ LayerNorm 参数：不应用权重衰减
- ❌ BatchNorm 参数：不应用权重衰减

### 混合精度训练

- 使用 `torch.amp.autocast` 自动转换精度
- 使用 `GradScaler` 防止梯度下溢
- 在梯度裁剪前需要 `unscale_` 梯度

## 文件清单

```
src/trainer/
├── tutorial/                           # 教学版本
│   ├── __init__.py                    # 模块初始化
│   ├── optimizer_from_scratch.py      # 手搓优化器
│   └── trainer_from_scratch.py        # 手搓训练器
├── optimizer.py                        # 精简优化器
├── trainer.py                          # 精简训练器（支持分布式）
├── checkpoint.py                       # 检查点管理
├── distributed_train_example.py       # 分布式训练示例
└── README.md                          # 模块文档
```

## 下一步计划

- [ ] 添加更多学习率调度策略
- [ ] 添加早停（Early Stopping）功能
- [ ] 添加 TensorBoard/WandB 日志支持
- [ ] 添加 FSDP (Fully Sharded Data Parallel) 支持
- [ ] 添加模型量化训练支持
- [ ] 添加更多的训练技巧示例

## 参考资源

- [Adam 论文](https://arxiv.org/abs/1412.6980)
- [AdamW 论文](https://arxiv.org/abs/1711.05101)
- [PyTorch DDP 教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)

