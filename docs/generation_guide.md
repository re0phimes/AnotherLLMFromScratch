# 生成功能使用指南

本指南介绍如何使用 AnotherLLMFromScratch 中的文本生成功能。

## 功能概述

项目现在支持以下生成功能：

1. **模型生成方法** (`GPT2Model.generate`)
   - 自回归文本生成
   - 多种采样策略（贪婪解码、top-k、top-p、temperature）
   - 自动序列长度管理

2. **训练时评估生成** (`BaseTrainer.generate_samples`)
   - 在训练过程中定期生成样本
   - 实时查看模型训练效果
   - 支持多个评估提示

## 1. 模型 generate 方法

### API 说明

```python
@torch.inference_mode()
def generate(
    self,
    input_ids: torch.Tensor,      # 输入 token ids，形状 (B, T)
    max_new_tokens: int,           # 要生成的最大 token 数量
    *,
    temperature: float = 1.0,      # 采样温度
    top_k: Optional[int] = None,   # Top-k 过滤
    top_p: Optional[float] = None, # Nucleus (top-p) 采样
    pad_token_id: Optional[int] = None,  # Padding token ID
) -> torch.Tensor:
```

### 采样策略

#### 1. 贪婪解码 (Greedy Decoding)
最确定性的生成方式，每次选择概率最高的 token。

```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.0,  # temperature=0 表示贪婪解码
)
```

#### 2. Temperature 采样
调整输出分布的随机性：
- `temperature > 1.0`: 更随机，更有创造性
- `temperature < 1.0`: 更确定，更保守
- `temperature = 1.0`: 标准采样

```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,  # 稍微降低随机性
)
```

#### 3. Top-k 采样
只从概率最高的 k 个 token 中采样。

```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,  # 从概率最高的 50 个 token 中采样
)
```

#### 4. Top-p (Nucleus) 采样
从累积概率达到 p 的最小 token 集合中采样。

```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,  # 从累积概率为 90% 的 token 集合中采样
)
```

#### 5. 组合策略（推荐）
结合 temperature、top-k 和 top-p 获得最佳效果。

```python
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    temperature=0.8,   # 降低随机性
    top_k=50,          # 限制候选集大小
    top_p=0.95,        # 进一步过滤低概率 token
)
```

### 完整使用示例

```python
import torch
from transformers import AutoTokenizer
from src.models.modeling_auto import AutoModelForCausalLM, AutoConfig

# 加载模型和 tokenizer
model_config = AutoConfig.from_dict({...})
model = AutoModelForCausalLM.from_config(model_config)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# 准备输入
prompt = "中国经济正处于"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# 生成
with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
    )

# 解码输出
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)
```

## 2. 训练时评估生成

### 配置说明

在训练配置文件中添加以下内容：

```yaml
training:
  max_epochs: 3
  eval_interval: 1000  # 每 1000 步生成一次样本

evaluation:
  prompts:
    - "中国经济正处于"
    - "在未来的科技创新领域"
    - "人工智能技术"
  max_new_tokens: 120
```

### 训练器使用

```python
from src.trainer.sft_trainer import SFTTrainer
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# 创建训练器
trainer = SFTTrainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    max_epochs=3,
    # ... 其他参数 ...
    
    # 生成评估配置
    eval_interval=1000,           # 每 1000 步生成一次
    eval_prompts=[                # 评估提示列表
        "中国经济正处于",
        "在未来的科技创新领域",
    ],
    eval_max_tokens=100,          # 每次生成 100 个 token
    eval_temperature=0.8,         # 采样温度
    eval_top_k=50,                # Top-k 采样
    eval_top_p=0.9,               # Top-p 采样
    tokenizer=tokenizer,          # 必须提供 tokenizer
)

# 开始训练（会自动定期生成样本）
trainer.train()
```

### 生成输出示例

训练过程中会看到如下输出：

```
======================================================================
生成样本 (Step 1000)
======================================================================

[Prompt 1]: 中国经济正处于
[Generated]: 中国经济正处于转型升级的关键时期，面临着诸多挑战和机遇...
----------------------------------------------------------------------

[Prompt 2]: 在未来的科技创新领域
[Generated]: 在未来的科技创新领域，人工智能、量子计算等前沿技术...
----------------------------------------------------------------------
```

## 3. 性能优化建议

### 生成速度优化

1. **使用 KV Cache**（未来功能）
   - 缓存已计算的键值对
   - 大幅提升生成速度

2. **批量生成**
   - 同时处理多个 prompt
   - 充分利用 GPU 并行能力

3. **减少序列长度**
   - 较短的 `max_new_tokens` 更快
   - 根据需求平衡长度和速度

### 内存优化

1. **使用 `@torch.inference_mode()`**
   - 已在代码中实现
   - 禁用梯度计算，节省内存

2. **限制 batch size**
   - 生成时使用较小的 batch
   - 避免内存溢出

## 4. 常见问题

### Q: 生成的文本质量不好？
A: 模型需要充分训练。在训练早期，生成结果可能是乱码。继续训练后质量会逐渐提升。

### Q: 生成速度很慢？
A: 
- 自回归生成本质上是顺序的，较慢是正常的
- 使用 GPU 可以显著加速
- 考虑减少 `max_new_tokens`

### Q: 如何选择最佳采样参数？
A: 推荐组合：
- **平衡质量和多样性**: `temperature=0.8, top_p=0.9`
- **更保守**: `temperature=0.6, top_k=40`
- **更随机**: `temperature=1.0, top_p=0.95`

### Q: 训练时不想生成样本？
A: 
- 不设置 `eval_interval` 参数
- 或设置 `eval_prompts=None`

## 5. 完整示例

查看以下示例文件：

- `test_generate.py`: 基础生成功能测试
- `examples/train_with_generation.py`: 带生成评估的完整训练流程

运行测试：
```bash
# 测试生成功能
.venv/bin/python test_generate.py

# 带生成评估的训练
.venv/bin/python examples/train_with_generation.py
```

## 6. 未来改进

计划添加的功能：

- [ ] KV Cache 支持（提升生成速度）
- [ ] Beam Search（提升生成质量）
- [ ] 流式生成（实时输出）
- [ ] 批量生成优化
- [ ] 更多采样策略（typical sampling, mirostat 等）
