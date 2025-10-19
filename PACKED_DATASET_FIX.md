# Packed Dataset 修复方案

## 修复 src/dataset/pretrain.py

### 方案 A：添加 Document Boundary Tokens（推荐）

在每个文档之间添加特殊的边界标记，并在 attention mask 中正确处理。

#### 1. 修改 PackedIterableDataset，记录文档边界

```python
class PackedIterableDataset(IterableDataset[Dict[str, Any]]):
    """迭代式 packed 预训练数据集，支持文档边界标记"""
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import random
        
        rng = random.Random(self.seed)
        token_buffer: Deque[int] = deque()
        shuffle_buffer: List[Dict[str, Any]] = []
        
        # ✅ 添加：记录每个 position 是否是文档边界（EOS token 位置）
        boundary_buffer: Deque[bool] = deque()
        
        for source in self.sources:
            dataset: IterableDataset
            if source.type == "local":
                dataset = LocalJsonlDataset(source)
            elif source.type == "streaming":
                dataset = StreamingIterableDataset(source)
            else:
                raise ValueError(f"Unknown source type: {source.type}")
            
            for record in dataset:
                shuffle_buffer.append(record)
                
                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                    rng.shuffle(shuffle_buffer)
                    
                    for rec in shuffle_buffer:
                        text = rec["text"]
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        tokens.append(self.eos_token_id)
                        
                        # ✅ 添加：标记文档边界
                        token_buffer.extend(tokens)
                        boundary_buffer.extend([False] * (len(tokens) - 1) + [True])
                    
                    shuffle_buffer.clear()
                    
                    while len(token_buffer) >= self.sequence_length:
                        chunk = [token_buffer.popleft() for _ in range(self.sequence_length)]
                        boundaries = [boundary_buffer.popleft() for _ in range(self.sequence_length)]
                        
                        yield {
                            "input_ids": torch.tensor(chunk, dtype=torch.long),
                            "boundaries": torch.tensor(boundaries, dtype=torch.bool),  # ✅ 新增
                            "source": "packed"
                        }
        
        # 处理剩余数据...
        if shuffle_buffer:
            rng.shuffle(shuffle_buffer)
            for rec in shuffle_buffer:
                text = rec["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.append(self.eos_token_id)
                token_buffer.extend(tokens)
                boundary_buffer.extend([False] * (len(tokens) - 1) + [True])
            shuffle_buffer.clear()
        
        while len(token_buffer) >= self.sequence_length:
            chunk = [token_buffer.popleft() for _ in range(self.sequence_length)]
            boundaries = [boundary_buffer.popleft() for _ in range(self.sequence_length)]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "boundaries": torch.tensor(boundaries, dtype=torch.bool),
                "source": "packed"
            }
```

#### 2. 修改 collate_fn，正确处理 attention mask

```python
def collate_fn(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if self.extras.pack_sequences and examples and "input_ids" in examples[0]:
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        boundaries = torch.stack([ex["boundaries"] for ex in examples])  # ✅ 获取边界信息
        
        # ✅ 构建正确的 attention mask
        # 方法1：完全阻断跨文档 attention
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        
        # 为每个 batch 构建 document-aware causal mask
        # 这需要在模型中处理，这里只是标记 padding
        # attention_mask 保持为 1（因为 packed 没有 padding）
        
        labels = input_ids.clone()
        
        # ✅ 在文档边界（EOS token）位置，不计算 loss
        # 这样模型不会过度学习 EOS 到下一个文档的模式
        labels[boundaries] = -100
        
        source_names = [ex.get("source", "packed") for ex in examples]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "boundaries": boundaries,  # ✅ 传递给模型，供 attention 使用
            "metadata": {"source": source_names},
        }
    
    # 原有的非 packed 逻辑...
    texts = [str(ex["text"]) for ex in examples]
    source_names = [str(ex.get("source", "unknown")) for ex in examples]

    padding = self._resolve_padding_strategy()
    encoded = self.tokenizer(
        texts,
        add_special_tokens=True,
        max_length=self.sequence_length,
        truncation=True,
        padding=padding,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "metadata": {"source": source_names},
    }
```

### 方案 B：简化方案 - 只在 EOS 位置设置 -100

如果不想修改模型的 attention 机制，可以先采用简化方案：

```python
def collate_fn(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if self.extras.pack_sequences and examples and "input_ids" in examples[0]:
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        # ✅ 关键修复：在 EOS token 位置不计算 loss
        # 假设 EOS token id 是已知的
        eos_mask = (input_ids == self.tokenizer.eos_token_id)
        labels[eos_mask] = -100
        
        source_names = [ex.get("source", "packed") for ex in examples]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": {"source": source_names},
        }
    # ... 其余代码不变
```

## 建议的实施步骤

### 立即修复（紧急）

1. **停止当前训练**
2. **使用 `gpt2_pretrain_fixed.yaml`** 配置（禁用 pack_sequences）
3. **删除旧 checkpoint**，重新开始训练
4. **监控生成质量**，确认问题解决

### 长期修复（优化）

1. **实施方案 A**，正确处理文档边界
2. **添加单元测试**，验证 packed dataset 的正确性
3. **重新启用 pack_sequences**，享受更高的训练效率

## 验证方法

在修复后，应该观察到：

1. **Loss 恢复正常**：2-5 之间
2. **PPL 恢复正常**：3-10 之间  
3. **生成文本质量提升**：无重复，语义连贯
4. **生成示例**：

```
新华社北京6月28日电 记者今日从国家统计局获悉...
（应该是完整、连贯的句子，而非重复词汇）
```

## 其他建议

### 1. 增加数据多样性

```yaml
data:
  data_sources:
    - path: /path/to/chinanews.jsonl
      type: local
      name: chinanews
      sampling_weight: 1.0
    - path: /path/to/wikipedia.jsonl
      type: local
      name: wikipedia
      sampling_weight: 0.5
    - path: /path/to/books.jsonl
      type: local
      name: books
      sampling_weight: 0.3
```

### 2. 添加 Early Stopping

监控验证集 loss，当开始上升时停止训练。

### 3. 定期检查生成质量

每隔几百步就运行生成评估，及早发现问题。
