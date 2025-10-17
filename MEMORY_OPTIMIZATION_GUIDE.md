# 显存优化指南

## 问题诊断

遇到 `CUDA out of memory` 错误时，说明 GPU 显存不足。

### 内存计算公式

对于 Transformer 模型，训练时的显存占用主要包括：

```
总显存 = 模型参数 + 优化器状态 + 梯度 + 激活值 + 中间缓存

详细分解：
1. 模型参数 (FP32):      125M × 4 bytes ≈ 0.5 GB
2. 模型参数 (FP16/BF16):  125M × 2 bytes ≈ 0.25 GB
3. 优化器状态 (AdamW):    125M × 12 bytes ≈ 1.5 GB
4. 梯度:                  125M × 4 bytes ≈ 0.5 GB
5. 激活值:                batch_size × seq_len × hidden × layers × factor
   - 对于 bs=8, seq=1024, hidden=768, layers=12
   - 约 8 × 1024 × 768 × 12 × 4-6 ≈ 12-18 GB
```

**关键点：激活值是显存使用的大头，与 batch_size 和 sequence_length 成正比。**

## 快速解决方案

### 方案 1：减小 micro_batch_size（推荐）✅

**最有效且最简单的方法**

修改配置：
```yaml
training:
  micro_batch_size: 4      # 从 8 降至 4（减少 50% 激活值内存）
  gradient_accumulation: 8  # 从 4 增至 8（保持有效 batch size 不变）
  # 有效 batch size = 4 × 8 = 32（与原来相同）
```

**优点：**
- ✅ 立即生效
- ✅ 不影响训练效果（有效 batch size 不变）
- ✅ 减少约 50% 激活值内存

**缺点：**
- ⚠ 训练速度稍慢（更多梯度累积步骤）

---

### 方案 2：减小 sequence_length

如果方案 1 仍然 OOM，可进一步减小序列长度：

```yaml
data:
  sequence_length: 512  # 从 1024 降至 512（减少 50% 激活值内存）

training:
  micro_batch_size: 4   # 或者可以恢复到 8
  gradient_accumulation: 8
```

**优点：**
- ✅ 显著减少内存使用（50%）
- ✅ 训练速度更快

**缺点：**
- ❌ 减少了上下文长度（可能影响长文本理解）
- ❌ 需要重新评估模型效果

---

### 方案 3：极度保守配置

如果显存仍然不够，使用最保守配置：

```yaml
data:
  sequence_length: 512

training:
  micro_batch_size: 2    # 最小值
  gradient_accumulation: 16
  # 有效 batch size = 2 × 16 = 32
```

**显存需求：** ~8-12 GB（适合 16GB 显卡）

---

### 方案 4：启用内存碎片优化

在运行前设置环境变量（如错误提示所建议）：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_sft_training.py --config configs/train/gpt2_pretrain_packed.yaml
```

**优点：**
- ✅ 减少内存碎片
- ✅ 可能额外节省 1-2 GB

---

### 方案 5：减少 DataLoader workers

```yaml
data:
  num_workers: 0  # 或 1-2（默认可能是 4）
```

DataLoader workers 会在 CPU 内存中预加载数据，但也会占用少量 GPU 内存用于固定内存 (pinned memory)。

---

## 配置对比表

| 配置 | micro_bs | grad_accum | seq_len | 有效 bs | 预计显存 | 适用显卡 |
|------|----------|------------|---------|---------|---------|---------|
| **原始** | 8 | 4 | 1024 | 32 | ~22-24 GB | 24GB+ ❌ OOM |
| **优化v1** | 4 | 8 | 1024 | 32 | ~14-16 GB | 16-24GB ✅ |
| **优化v2** | 4 | 8 | 512 | 32 | ~10-12 GB | 12-16GB ✅ |
| **保守** | 2 | 16 | 512 | 32 | ~8-10 GB | 8-12GB ✅ |

---

## 逐步调试流程

### 步骤 1：尝试中等优化配置

```bash
cd /home/modelenv/chentianxuan/projects/llm/AnotherLLMFromScratch
source .venv/bin/activate

# 使用修改后的配置（micro_batch_size=4）
python scripts/run_sft_training.py \
    --config configs/train/gpt2_pretrain_packed.yaml
```

**期望：** 训练成功启动，显存使用 ~14-16 GB

---

### 步骤 2：如果仍然 OOM，使用低内存配置

```bash
# 使用极度保守的配置（micro_batch_size=2）
python scripts/run_sft_training.py \
    --config configs/train/gpt2_pretrain_packed_lowmem.yaml
```

**期望：** 训练成功启动，显存使用 ~12-14 GB

---

### 步骤 3：监控显存使用

训练运行时，另开终端监控：

```bash
watch -n 1 nvidia-smi
```

观察：
- **GPU Memory-Usage**：应该在 18-20 GB 以下（留 2-4GB 缓冲）
- **Volatile GPU-Util**：应该接近 100%（GPU 利用率）

---

## 高级优化（如果需要）

### 选项 1：启用 Gradient Checkpointing

**原理：** 以计算时间换显存空间（减少 40-60% 激活值内存）

需要在模型代码中添加：
```python
# src/models/gpt2/model.py
def __init__(self, config):
    # ...
    self.gradient_checkpointing = False  # 添加此标志
    
def forward(self, ...):
    if self.gradient_checkpointing and self.training:
        # 使用 checkpoint 包装 transformer blocks
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)
```

**权衡：**
- ✅ 减少 40-60% 激活值显存
- ❌ 训练速度慢 20-30%（需要重新计算前向传播）

---

### 选项 2：混合精度训练优化

确保配置中：
```yaml
training:
  use_amp: true  # 必须启用！FP16 可减少 50% 显存
```

检查是否正确启用：
```python
# 训练日志应该显示
SFTTrainer 配置
混合精度: True  ← 确认此项为 True
```

---

### 选项 3：使用更小的模型

如果最终仍无法训练 125M 模型，考虑创建更小的配置：

```yaml
# configs/model/gpt_60m.yaml
vocab_size: 151646
n_layer: 8        # 从 12 降至 8
n_head: 8         # 从 12 降至 8
n_embd: 512       # 从 768 降至 512
block_size: 1024
```

**显存需求：** ~6-8 GB（60M 参数）

---

## 最佳实践建议

### ✅ 推荐做法

1. **从保守配置开始**：先用 `micro_batch_size=2` 验证能否训练
2. **逐步增加**：如果显存充足，逐步增加到 4、8
3. **监控显存**：始终保留 2-4 GB 缓冲空间
4. **启用 AMP**：混合精度训练是必须的
5. **保持有效 batch size**：通过梯度累积保持在 32-64

### ❌ 避免做法

1. **不要一次性调太多参数**：逐步调整，找到最优点
2. **不要禁用 AMP**：这会直接翻倍显存使用
3. **不要忽略显存监控**：OOM 前通常有警告信号
4. **不要过度减小 batch size**：micro_batch_size < 2 可能导致训练不稳定

---

## 当前建议行动

### 立即执行：

1. **已修改主配置**：`gpt2_pretrain_packed.yaml` 
   - micro_batch_size: 8 → 4
   - gradient_accumulation: 4 → 8

2. **尝试运行**：
   ```bash
   cd /home/modelenv/chentianxuan/projects/llm/AnotherLLMFromScratch
   source .venv/bin/activate
   
   # 先尝试优化后的主配置
   python scripts/run_sft_training.py \
       --config configs/train/gpt2_pretrain_packed.yaml
   ```

3. **如果仍然 OOM，使用低内存配置**：
   ```bash
   python scripts/run_sft_training.py \
       --config configs/train/gpt2_pretrain_packed_lowmem.yaml
   ```

4. **监控显存**：
   ```bash
   # 另开终端
   watch -n 1 nvidia-smi
   ```

---

## 预期结果

### 成功标志 ✅
```
训练日志显示：
Epoch 0 | Step 10  | Batch 20/?  | Loss: 10.x | PPL: xxx | ...
Epoch 0 | Step 20  | Batch 40/?  | Loss: 9.x  | PPL: xxx | ...
...

nvidia-smi 显示：
GPU Memory-Usage: 14-18 GB / 24 GB
Volatile GPU-Util: 90-100%
```

### 失败标志 ❌
```
torch.OutOfMemoryError: CUDA out of memory...
```

如果仍然失败，需要：
1. 使用 `gpt2_pretrain_packed_lowmem.yaml`
2. 或进一步减小 sequence_length 到 512
3. 或实现 gradient checkpointing

---

## 总结

**记住：显存优化的核心是找到平衡点**

| 参数 | 影响 | 优先级 |
|------|------|--------|
| micro_batch_size | ⭐⭐⭐⭐⭐ 最大影响 | 🔥 最先调整 |
| sequence_length | ⭐⭐⭐⭐ 较大影响 | 🔥 次要调整 |
| gradient_accumulation | ⭐ 无影响（仅速度） | ✅ 用于补偿 |
| use_amp | ⭐⭐⭐⭐⭐ 必须启用 | 🔥 必选项 |
| gradient_checkpointing | ⭐⭐⭐ 显著影响 | 🔧 高级选项 |
