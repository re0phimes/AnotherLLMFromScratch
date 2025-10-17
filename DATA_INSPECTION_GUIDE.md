# 数据检查功能使用指南

## 🎯 功能说明

在训练开始前自动检查数据处理流程，确保数据正确性。检查内容包括：

1. **原始数据结构** - Batch keys, shapes
2. **Token 统计** - PAD, EOS, BOS tokens 数量
3. **Attention Mask** - 值分布检查
4. **Labels** - 有效值和忽略值 (-100) 统计
5. **样本详细内容** - Token IDs、解码文本、跨文档检查

---

## ✅ 启用方式

### 自动启用（默认）

训练时会自动检查第一个 batch：

```bash
cd /home/modelenv/chentianxuan/projects/llm/AnotherLLMFromScratch
source .venv/bin/activate

# 默认会自动检查数据
python scripts/run_sft_training.py \
    --config configs/train/gpt2_pretrain_packed.yaml
```

### 禁用检查

如果不想检查（例如恢复训练），可以禁用：

```bash
python scripts/run_sft_training.py \
    --config configs/train/gpt2_pretrain_packed.yaml \
    --no-inspect-data
```

### 独立测试脚本

不启动训练，只检查数据：

```bash
python test_data_inspection.py
```

---

## 📋 输出示例

### 1. Batch 结构信息

```
================================================================================
📋 BATCH 数据检查
================================================================================

1️⃣  Batch 结构:
   Keys: ['input_ids', 'attention_mask', 'labels', 'metadata']

2️⃣  张量形状:
   input_ids shape:      torch.Size([4, 1024])
   attention_mask shape: torch.Size([4, 1024])
   labels shape:         torch.Size([4, 1024])

   📊 Batch size: 4
   📏 Sequence length: 1024
```

### 2. Token 统计

```
3️⃣  Token 统计:
   Tokenizer 特殊 tokens:
      PAD token ID: 151643
      EOS token ID: 151643
      BOS token ID: 151644

   PAD tokens: 0/4096 (0.00%)
   EOS tokens: 8/4096 (0.20%)
   BOS tokens: 0/4096 (0.00%)
```

**✅ 好的情况：**
- PAD tokens 0% - 打包模式正确，无填充
- EOS tokens 0.2% - 有文档分隔符
- BOS tokens 0% - 打包模式不需要每个样本都有 BOS

**❌ 坏的情况：**
- PAD tokens > 50% - 说明有大量填充，浪费算力
- EOS tokens 0% - 可能没有正确添加分隔符

### 3. Attention Mask

```
4️⃣  Attention Mask:
   值为 1: 4096/4096 (100.00%)
   值为 0: 0/4096 (0.00%)
```

**✅ 打包模式应该全为 1**（无 padding）
**❌ 如果有 0，说明有 padding，检查配置**

### 4. Labels

```
5️⃣  Labels:
   有效 labels: 4096/4096 (100.00%)
   忽略 labels (-100): 0/4096 (0.00%)
   ✅ 有效 labels 与 input_ids 完全匹配
```

**✅ 打包模式应该：**
- 有效 labels 100%（无 -100）
- Labels 与 input_ids 完全匹配

**❌ 传统模式可能有 -100**（padding 位置）

### 5. 样本详细内容

```
================================================================================
7️⃣  样本详细内容 (显示前 2 个样本)
================================================================================

📄 样本 #1:
--------------------------------------------------------------------------------

   A. Token IDs (前 50 个):
      [104116, 109904, 100910, 151643, 23845, 100307, ...]

   B. Attention Mask (前 50 个):
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]

   C. Labels (前 50 个):
      [104116, 109904, 100910, 151643, 23845, 100307, ...]

   D. EOS Token 位置 (前 5 个): [3, 18, 45, 67, 89]

   E. 解码后的文本:
      前 50 tokens: '新华社北京<|endoftext|>据悉，该项目...'

      第一个文档段落 (到第一个 EOS):
      '新华社北京1月15日电（记者张晓松）中国政府今天宣布...'

   F. 跨文档检查:
      ✅ 此样本包含 5 个文档 (检测到 5 个 EOS tokens)

      文档 1 (3 tokens):
      '新华社北京'

      文档 2 (15 tokens):
      '据悉，该项目将于今年年底前完成...'
```

**关键检查点：**

✅ **Token IDs 正常**：包含各种中文 token
✅ **Attention Mask 全为 1**：无 padding
✅ **Labels 与 input_ids 相同**：正确的预训练设置
✅ **有 EOS tokens**：文档分隔符存在
✅ **跨文档**：样本包含多个文档片段
✅ **解码文本合理**：中文新闻内容

❌ **需要注意：**
- 如果解码失败或乱码，检查 tokenizer
- 如果没有 EOS tokens，检查打包逻辑
- 如果全是同一个文档，检查 shuffle buffer

---

## 🔍 常见问题排查

### 问题 1：大量 PAD tokens

**症状：**
```
PAD tokens: 2048/4096 (50.00%)
```

**原因：** 使用了传统模式（padding），而非打包模式

**解决：**
```yaml
# 确保配置中启用打包
data:
  pack_sequences: true
```

### 问题 2：没有 EOS tokens

**症状：**
```
EOS tokens: 0/4096 (0.00%)
D. EOS Token 位置 (前 5 个): []
```

**原因：** `PackedPretrainDataset` 可能没有正确添加 EOS

**解决：** 检查 `src/dataset/pretrain.py` 中的逻辑：
```python
tokens = self.tokenizer.encode(text, add_special_tokens=False)
tokens.append(self.eos_token_id)  # 确保这行存在
```

### 问题 3：Labels 有 -100

**症状：**
```
忽略 labels (-100): 2048/4096 (50.00%)
```

**原因：** 使用了 SFT 模式的 collate_fn，而非 pretrain 模式

**解决：** 确保 `collate_fn` 检测到打包模式：
```python
if self.extras.pack_sequences and "input_ids" in examples[0]:
    # 打包模式：labels = input_ids，无 -100
    labels = input_ids.clone()
```

### 问题 4：Attention Mask 不全为 1

**症状：**
```
值为 1: 2048/4096 (50.00%)
值为 0: 2048/4096 (50.00%)
```

**原因：** 有 padding

**解决：** 同问题 1，启用打包模式

### 问题 5：解码乱码

**症状：**
```
解码后的文本: '������...'
```

**原因：** Tokenizer 不匹配

**解决：**
```bash
# 确认配置中的 tokenizer 正确
grep tokenizer_name_or_path configs/train/*.yaml

# 应该是 Qwen/Qwen2.5-0.5B
```

### 问题 6：只有一个文档（未跨文档）

**症状：**
```
✅ 此样本包含 1 个文档
```

**原因：** Shuffle buffer 太小，或数据源单一

**解决：**
```yaml
data:
  shuffle_buffer_size: 5000  # 增加到 5000+
```

---

## 🛠️ 高级用法

### 在代码中手动检查

```python
from src.utils import inspect_batch, inspect_first_batch
from transformers import AutoTokenizer

# 方法 1：检查 DataLoader 的第一个 batch
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
inspect_first_batch(dataloader, tokenizer, num_samples=3)

# 方法 2：检查单个 batch
batch = next(iter(dataloader))
inspect_batch(batch, tokenizer, max_samples=3, max_seq_display=100)
```

### 调整显示参数

```python
inspect_first_batch(
    dataloader, 
    tokenizer, 
    num_samples=5  # 显示 5 个样本（默认 2）
)

inspect_batch(
    batch, 
    tokenizer, 
    max_samples=3,         # 最多 3 个样本
    max_seq_display=200    # 每个样本显示 200 个 tokens（默认 50）
)
```

---

## 📊 验证清单

训练前确保：

- [ ] **Batch 结构正确** - 包含 input_ids, attention_mask, labels
- [ ] **形状匹配** - 所有张量形状一致
- [ ] **无过度 padding** - PAD tokens < 5%（理想 0%）
- [ ] **有文档分隔符** - EOS tokens > 0
- [ ] **Attention mask 正确** - 打包模式全为 1
- [ ] **Labels 正确** - 打包模式无 -100，且等于 input_ids
- [ ] **解码正常** - 文本可读，内容合理
- [ ] **跨文档** - 样本包含多个文档片段

全部通过 ✅ → 可以开始训练！

---

## 📁 相关文件

- **检查工具**: `src/utils/data_inspection.py`
- **测试脚本**: `test_data_inspection.py`
- **集成位置**: `scripts/run_sft_training.py`

---

## 🎓 工作原理

### 数据流程

```
原始 JSONL 文件
    ↓
LocalJsonlDataset 读取
    ↓
PackedPretrainDataset 打包
    ├─ Shuffle buffer (5000 samples)
    ├─ 拼接文本 + EOS
    ├─ Tokenize
    └─ 切片为固定长度 (1024)
    ↓
DataLoader 组批 (batch_size=4)
    ↓
Collate_fn 处理
    ├─ 堆叠 input_ids
    ├─ 生成 attention_mask (全1)
    └─ 生成 labels (= input_ids)
    ↓
Trainer 训练
```

### 检查时机

```python
# run_sft_training.py
artifacts = prepare_components(...)  # 创建 dataloader

# 👉 在这里插入检查
if is_main_process() and args.inspect_data:
    inspect_first_batch(artifacts.train_loader, artifacts.tokenizer)

trainer = SFTTrainer(...)  # 创建 trainer
trainer.train()  # 开始训练
```

---

## 🚀 快速开始

1. **使用默认设置启动训练**（自动检查）：
   ```bash
   python scripts/run_sft_training.py --config configs/train/gpt2_pretrain_packed.yaml
   ```

2. **查看输出**，确认数据正确

3. **如果有问题**，参考上面的"常见问题排查"

4. **如果一切正常**，训练会自动继续

---

## 💡 最佳实践

1. **首次训练必检查** - 确保数据处理正确
2. **修改配置后检查** - 验证变更效果
3. **新数据集必检查** - 不同数据源可能有问题
4. **恢复训练可跳过** - 使用 `--no-inspect-data`

**记住：花 1 分钟检查数据，可以避免浪费几小时训练！** 🎯
