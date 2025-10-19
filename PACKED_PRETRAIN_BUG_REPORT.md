# Packed Pretrain 重复生成问题诊断报告

## 问题现象

1. **严重的文本重复**：生成的文本出现大量重复词汇（"落后落后落后..."、"警察警察警察..."）
2. **异常低的 Loss 和 PPL**：Loss 0.01-0.06，PPL 1.01-1.07（正常应该在 3-10 之间）
3. **训练步数过高**：已达到 59510+ 步，可能严重过拟合

## 根本原因分析

### 1. **Packed Dataset 的 Attention Mask 错误** ⚠️ 核心问题

在 `src/dataset/pretrain.py` 第 334 行：

```python
attention_mask = torch.ones_like(input_ids)  # ❌ 所有位置都是 1
```

**问题**：
- Packed sequence 将多个文档拼接在一起，每个文档之间只有 EOS token 分隔
- 但 attention_mask 全部为 1，意味着模型会**跨文档边界进行 attention**
- 模型学习到了错误的跨文档依赖关系，导致生成时混乱和重复

**正确行为**：
- 每个文档应该有独立的 attention 边界
- 文档 A 的 token 不应该 attend 到文档 B 的 token
- 需要构建 document boundary mask

### 2. **Labels 处理不完整**

在 packed mode 下：

```python
labels = input_ids.clone()  # ❌ 没有任何 -100 标记
```

**问题**：
- 所有 token（包括 EOS token）都参与 loss 计算
- 模型过度学习了 EOS 到下一个文档开始的模式
- 导致生成时无法正确处理文档边界

### 3. **配置参数未生效**

配置文件中设置：
```yaml
add_bos: false
add_eos: false
```

但在 packed dataset 实现中（第 224、241 行）：
```python
tokens.append(self.eos_token_id)  # 硬编码添加 EOS
```

配置参数在 packed mode 下完全被忽略。

### 4. **过拟合问题**

- 单一数据源（只有 chinanews_pretrain.jsonl）
- 训练步数过多（59510+ 步）
- Loss 异常低，说明模型已经"记住"了训练数据

## 解决方案

### 方案 1：修复 Packed Dataset（推荐用于大规模训练）

需要修改 `src/dataset/pretrain.py` 的 collate_fn，添加正确的 document boundary attention mask。

### 方案 2：暂时禁用 Packed Sequences（快速修复）

修改配置文件，使用传统的单文档训练方式：

```yaml
data:
  pack_sequences: false  # 改为 false
  add_bos: true          # 改为 true，添加文档开始标记
  add_eos: true          # 改为 true，添加文档结束标记
  padding: longest       # 改为 longest 或 max_length
```

### 方案 3：重新开始训练

由于当前模型已经严重过拟合并学习到错误模式：

1. **删除旧的 checkpoint**：`./checkpoints/gpt2_pretrain_packed_lowmem/`
2. **使用修复后的配置重新训练**
3. **监控指标**：
   - Loss 应该在 2-5 之间
   - PPL 应该在 3-10 之间
   - 定期检查生成质量

## 推荐配置

我将创建一个修复后的配置文件：`gpt2_pretrain_fixed.yaml`
