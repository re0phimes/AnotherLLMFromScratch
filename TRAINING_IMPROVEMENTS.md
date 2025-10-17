# 训练改进：ETA 显示 & 生成评估

## 🎯 解决的问题

### 问题 1：ETA 显示为 N/A
**原因**：IterableDataset 无法获取总长度，导致无法计算剩余时间。

**解决方案**：使用估算的 `steps_per_epoch` 来计算 ETA。

### 问题 2：没有定期生成评估
**原因**：配置文件缺少 `evaluation` 部分。

**解决方案**：添加评估配置，包含示例 prompts。

---

## ✅ 实现的改进

### 1. ETA 计算增强

**修改文件：`src/trainer/base_trainer.py`**

#### 新增参数
```python
def __init__(
    self,
    # ... 其他参数
    estimated_steps_per_epoch: Optional[int] = None,  # 新增
    **kwargs
):
    self._estimated_steps_per_epoch = estimated_steps_per_epoch
```

#### 改进的 ETA 逻辑
```python
# 计算 ETA
if self._epoch_total_batches is not None:
    # 情况 1：精确已知总批次数（常规 Dataset）
    remaining_batches = max(self._epoch_total_batches - processed_batches, 0)
    eta_seconds = remaining_batches * avg_time
    
elif self._estimated_steps_per_epoch is not None:
    # 情况 2：使用估算的 steps（IterableDataset）
    current_step_in_epoch = self.global_step % self._estimated_steps_per_epoch
    remaining_steps = max(self._estimated_steps_per_epoch - current_step_in_epoch, 0)
    remaining_batches = remaining_steps * self.grad_accum_steps
    eta_seconds = remaining_batches * avg_time
    
else:
    # 情况 3：无任何信息
    eta_seconds = 0.0
```

#### 改进的日志格式

**原来（ETA 为 N/A）：**
```
Epoch 0 | Step 18300 | Batch 146400/? | Loss: 3.41 | PPL: 30.28 | ETA: N/A
```

**现在（显示估算 ETA）：**
```
Epoch 0 | Step 300 | Step 300/1000 in epoch | Loss: 3.41 | PPL: 30.28 | ETA: 01:23:45
```

---

### 2. 生成评估配置

**修改文件：**
- `configs/train/gpt2_pretrain_packed.yaml`
- `configs/train/gpt2_pretrain_packed_lowmem.yaml`

#### 新增配置部分

```yaml
training:
  eval_interval: 500  # 每 500 步进行生成评估

# 生成评估配置
evaluation:
  prompts:
    - "新华社北京"
    - "据悉，"
    - "记者从"
    - "今日，"
    - "中国"
  max_new_tokens: 100
  temperature: 0.8
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
```

#### 效果

**每 500 步会输出生成结果：**
```
======================================================================
生成评估 @ Step 500
======================================================================

Prompt: 新华社北京
Generated: 新华社北京1月15日电（记者张晓松）中国政府今天宣布...

Prompt: 据悉，
Generated: 据悉，该项目将于今年年底前完成...

Prompt: 记者从
Generated: 记者从有关部门获悉，目前相关工作正在有序推进...

======================================================================
```

---

### 3. 训练脚本更新

**修改文件：`scripts/run_sft_training.py`**

#### 新增 `steps_per_epoch` 追踪

```python
@dataclass
class TrainingArtifacts:
    # ... 其他字段
    steps_per_epoch: Optional[int] = None  # 新增

# 在 prepare_components 中
if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
    max_steps = training_cfg.get("max_steps")
    if max_steps is None:
        steps_per_epoch = 1000  # 默认估算值
        logger.warning("⚠ IterableDataset 无法确定长度，使用默认 steps_per_epoch={}", steps_per_epoch)
    else:
        steps_per_epoch = max_steps // max_epochs
else:
    steps_per_epoch = len(train_loader)

return TrainingArtifacts(
    # ...
    steps_per_epoch=steps_per_epoch if isinstance(...) else None,
)

# 创建 trainer 时传递
trainer = SFTTrainer(
    # ...
    estimated_steps_per_epoch=artifacts.steps_per_epoch,
    # ...
)
```

---

## 📊 效果对比

### 修复前 ❌

```
Epoch 0 | Step 18270 | Batch 146160/? | Loss: 3.64 | PPL: 37.98 | ETA: N/A
Epoch 0 | Step 18280 | Batch 146240/? | Loss: 3.67 | PPL: 39.24 | ETA: N/A
Epoch 0 | Step 18290 | Batch 146320/? | Loss: 3.44 | PPL: 31.34 | ETA: N/A

问题：
- ❌ Batch 数字太大（146400），看不出进度
- ❌ ETA 显示 N/A，不知道还要多久
- ❌ 没有生成评估输出
```

### 修复后 ✅

```
Epoch 0 | Step 270 | Step 270/1000 in epoch | Loss: 3.64 | PPL: 37.98 | ETA: 02:15:30
Epoch 0 | Step 280 | Step 280/1000 in epoch | Loss: 3.67 | PPL: 39.24 | ETA: 02:14:45
Epoch 0 | Step 290 | Step 290/1000 in epoch | Loss: 3.44 | PPL: 31.34 | ETA: 02:14:00

...

======================================================================
生成评估 @ Step 500
======================================================================

Prompt: 新华社北京
Generated: 新华社北京1月15日电 记者从国家统计局获悉，2024年...
(50 tokens)

Prompt: 据悉，
Generated: 据悉，本次会议将就当前经济形势进行深入讨论...
(48 tokens)

======================================================================

优势：
- ✅ 清晰显示当前进度（270/1000）
- ✅ 实时估算剩余时间（ETA: 02:15:30）
- ✅ 定期输出生成样本，监控训练质量
```

---

## 🔧 配置调整建议

### 调整评估频率

**更频繁评估（适合调试）：**
```yaml
training:
  eval_interval: 100  # 每 100 步
```

**较少评估（适合长时间训练）：**
```yaml
training:
  eval_interval: 1000  # 每 1000 步
```

### 调整生成参数

**更保守的生成（减少重复）：**
```yaml
evaluation:
  temperature: 0.7        # 降低随机性
  repetition_penalty: 1.2  # 增加惩罚
  top_k: 40               # 减少候选
```

**更多样化的生成：**
```yaml
evaluation:
  temperature: 0.9        # 增加随机性
  repetition_penalty: 1.0  # 减少惩罚
  top_p: 0.95             # 增加候选范围
```

### 添加更多 prompts

```yaml
evaluation:
  prompts:
    - "新华社北京"
    - "据悉，"
    - "记者从"
    - "今日，"
    - "中国"
    - "本报讯"          # 新增
    - "根据最新消息"     # 新增
    - "专家表示"        # 新增
    - "数据显示"        # 新增
    - "会议指出"        # 新增
```

---

## 📈 使用方法

### 启动训练（自动使用改进）

```bash
cd /home/modelenv/chentianxuan/projects/llm/AnotherLLMFromScratch
source .venv/bin/activate

# 使用标准配置（已包含 evaluation）
python scripts/run_sft_training.py \
    --config configs/train/gpt2_pretrain_packed.yaml

# 或使用低内存配置
python scripts/run_sft_training.py \
    --config configs/train/gpt2_pretrain_packed_lowmem.yaml
```

### 查看训练日志

```bash
# 实时查看
tail -f checkpoints/logs/train_*.log

# 或使用 grep 过滤生成评估
tail -f checkpoints/logs/train_*.log | grep -A 20 "生成评估"
```

---

## 🎓 技术细节

### ETA 计算原理

对于 IterableDataset：

```python
# 假设参数
steps_per_epoch = 1000          # 估算每个 epoch 的步数
current_step = 270              # 当前全局步数
grad_accum_steps = 8            # 梯度累积步数

# 计算当前 epoch 的进度
current_step_in_epoch = 270 % 1000 = 270

# 计算剩余步数
remaining_steps = 1000 - 270 = 730

# 计算剩余批次数（考虑梯度累积）
remaining_batches = 730 × 8 = 5840

# 计算平均每批次时间
avg_time = elapsed_time / processed_batches

# 估算 ETA
eta_seconds = remaining_batches × avg_time
```

### 生成评估原理

```python
# 每 eval_interval 步触发
if self.global_step % self.eval_interval == 0:
    for prompt in eval_prompts:
        # Tokenize
        input_ids = tokenizer.encode(prompt)
        
        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids)
        
        # Log
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated_text}")
```

---

## 🔍 故障排除

### ETA 仍然显示 N/A

**可能原因：**
1. 使用旧版本代码（未更新 base_trainer.py）
2. `estimated_steps_per_epoch` 未正确传递

**解决方案：**
```bash
# 检查是否使用最新代码
grep "estimated_steps_per_epoch" src/trainer/base_trainer.py

# 重启训练
pkill -f run_sft_training.py
python scripts/run_sft_training.py --config ...
```

### 没有生成评估输出

**可能原因：**
1. 配置文件缺少 `evaluation` 部分
2. `eval_interval` 设置太大，还未触发

**解决方案：**
```bash
# 检查配置
grep -A 10 "evaluation:" configs/train/gpt2_pretrain_packed.yaml

# 确认包含 prompts
# 如果没有，添加上面提供的配置
```

### 生成质量差

**调整策略：**
1. **重复问题**：增加 `repetition_penalty` 到 1.2-1.5
2. **无意义输出**：降低 `temperature` 到 0.6-0.7
3. **太保守**：增加 `temperature` 到 0.9-1.0
4. **训练不足**：继续训练，loss 需要降至 3 以下

---

## 📚 相关文件

修改的文件：
- ✅ `src/trainer/base_trainer.py` - ETA 计算逻辑
- ✅ `scripts/run_sft_training.py` - 参数传递
- ✅ `configs/train/gpt2_pretrain_packed.yaml` - 评估配置
- ✅ `configs/train/gpt2_pretrain_packed_lowmem.yaml` - 评估配置

新增文档：
- 📄 `TRAINING_IMPROVEMENTS.md` - 本文档

---

## 🎉 总结

**两个关键改进：**

1. **ETA 显示** 
   - ✅ 基于估算 steps 计算剩余时间
   - ✅ 清晰显示训练进度
   - ✅ 更好的用户体验

2. **生成评估**
   - ✅ 定期输出生成样本
   - ✅ 实时监控模型质量
   - ✅ 及早发现问题（如重复、崩溃等）

**现在训练日志更加信息丰富，便于监控和调试！** 🚀
