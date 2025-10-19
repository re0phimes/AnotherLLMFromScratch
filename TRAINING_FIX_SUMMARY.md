# 训练重复问题修复总结

## 问题诊断

经过全面排查，发现以下关键问题导致模型生成严重重复：

### 1. **Packed Dataset 的 Labels 处理错误** ⚠️
- **原因**：在 `src/dataset/pretrain.py` 中，packed mode 下所有 token（包括 EOS）都参与 loss 计算
- **后果**：模型过度学习了"文档A的EOS → 文档B的开始"这种虚假的跨文档依赖
- **修复**：✅ 已在 EOS token 位置设置 `labels = -100`，不计算 loss

### 2. **Attention Mask 的理论问题** ⚠️
- **原因**：所有位置的 attention_mask 都是 1，模型会跨文档边界进行 attention
- **影响**：虽然这会导致一些跨文档信息泄露，但主要问题还是 labels 处理
- **长期方案**：需要实现 document-aware attention mask（见 PACKED_DATASET_FIX.md）

### 3. **过拟合问题**
- **训练步数**：59510+ 步，远超必要的训练量
- **Loss/PPL 异常低**：Loss 0.01-0.06，PPL 1.01-1.07（正常应该在 2-5 和 3-10）
- **数据单一**：只有一个数据源，容易过拟合

### 4. **配置问题**
- `add_bos: false` 和 `add_eos: false` 导致模型缺少序列边界信息
- 虽然 packed dataset 硬编码添加了 EOS，但缺少 BOS 仍然有影响

## 已实施的修复

### ✅ 代码修复
修改了 `src/dataset/pretrain.py` 的 `collate_fn` 方法：

```python
# 在 EOS token 位置不计算 loss
if self.tokenizer.eos_token_id is not None:
    eos_mask = (input_ids == self.tokenizer.eos_token_id)
    labels[eos_mask] = -100
```

### ✅ 创建了新配置
`configs/train/gpt2_pretrain_fixed.yaml`：
- 禁用了 pack_sequences（暂时）
- 启用了 add_bos 和 add_eos
- 调整了学习率和批次大小
- 增加了 repetition_penalty

## 推荐的行动方案

### 方案 A：立即重启训练（推荐）

当前模型已经严重过拟合，建议重新开始：

```bash
# 1. 备份旧 checkpoint（如需要）
mv checkpoints/gpt2_pretrain_packed_lowmem checkpoints/gpt2_pretrain_packed_lowmem.backup

# 2. 使用修复后的配置开始训练
python scripts/train_pretrain.py --config configs/train/gpt2_pretrain_fixed.yaml

# 3. 监控训练指标
# - Loss 应该在 2-5 之间
# - PPL 应该在 3-10 之间
# - 定期检查生成质量
```

### 方案 B：继续使用 packed sequences（需测试）

如果想继续使用 packed sequences 以提高效率：

```bash
# 1. 验证代码修复是否有效
python -m pytest tests/test_packed_dataset.py  # 如果有测试的话

# 2. 使用原配置，但重新开始训练
# 确保使用最新的代码（已修复 EOS labels 问题）
python scripts/train_pretrain.py --config configs/train/gpt2_pretrain_packed_lowmem.yaml
```

⚠️ **注意**：方案 B 仍然存在跨文档 attention 的理论问题，需要进一步验证效果。

## 训练监控指标

### 正常的训练指标
- **Loss**：2.0 - 5.0（初期可能更高）
- **Perplexity**：3.0 - 10.0
- **生成质量**：连贯、无明显重复

### 异常的训练指标（需警惕）
- ❌ Loss < 0.5：过拟合
- ❌ Loss 不下降：学习率问题或数据问题
- ❌ 生成重复：labels 处理问题或 repetition_penalty 不足

### 评估命令示例

```python
# 在训练脚本中定期运行生成评估
prompts = [
    "新华社北京",
    "据悉，",
    "记者从",
    "中国经济正处于",
]

for prompt in prompts:
    output = model.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.9,
        repetition_penalty=1.2
    )
    print(f"[Prompt]: {prompt}")
    print(f"[Generated]: {output}\n")
```

## 长期优化建议

### 1. 增加数据多样性

```yaml
data:
  data_sources:
    - path: /path/to/chinanews.jsonl
      name: chinanews
      sampling_weight: 1.0
    - path: /path/to/wikipedia.jsonl
      name: wikipedia
      sampling_weight: 0.5
```

### 2. 实施完整的 Document-Aware Attention

参考 `PACKED_DATASET_FIX.md` 中的方案 A，实现真正的文档边界 attention mask。

### 3. 添加验证集和 Early Stopping

```yaml
data:
  val_data_sources:
    - path: /path/to/val.jsonl
      name: validation

training:
  eval_steps: 500
  early_stopping_patience: 3
```

### 4. 使用更多的正则化

```yaml
training:
  dropout: 0.1          # 在模型配置中
  label_smoothing: 0.1  # 在训练配置中
```

## 预期效果

修复后应该看到：

### ✅ 生成示例（修复后）
```
[Prompt]: 新华社北京
[Generated]: 新华社北京6月28日电 记者今日从国家统计局获悉，今年前5个月，
全国规模以上工业企业实现利润总额同比增长8.5%，增速比前4个月回落0.3个
百分点。分析人士指出，当前工业企业经济运行总体平稳...
```

### ❌ 生成示例（修复前）
```
[Prompt]: 新华社北京  
[Generated]: 新华社北京落后落后落后落后落后恒大恒大恒大警察警察警察...
```

## 相关文档

- `PACKED_PRETRAIN_BUG_REPORT.md` - 详细的问题诊断报告
- `PACKED_DATASET_FIX.md` - 完整的代码修复方案
- `gpt2_pretrain_fixed.yaml` - 修复后的训练配置

## 需要帮助？

如果问题持续存在，检查：
1. ✅ 代码修复是否正确应用
2. ✅ 是否使用了新的 checkpoint（而非旧的过拟合模型）
3. ✅ 生成时的 repetition_penalty 是否足够（建议 1.2-1.5）
4. ✅ tokenizer 的 eos_token_id 是否正确设置
