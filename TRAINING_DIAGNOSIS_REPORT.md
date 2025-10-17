# 训练问题诊断报告

## 问题描述

训练完成后，模型生成全是重复token：
```
生成样本 (Step 18000)
[Generated]: 中国经济正处于正处于正处于正处于正处于...
[Generated]: 在未来的科技创新领域科技创新领域领域科技创新领域...
[Generated]: china is one the age of one of of one of of of of...
```

训练日志显示：
- **Loss = 0.0000, PPL = 1.00** (严重异常)
- 正常的语言模型Loss应在2-4之间，PPL在10-50之间

---

## 诊断结果

### ✅ 数据处理正常

运行`debug_data_pipeline.py`验证：
- Tokenization正常
- Labels设置正确
- 有效label比例：82.57% (健康)
- 数据内容质量良好

**结论：问题不在数据预处理**

---

### ❌ 核心问题：严重过拟合 + 配置不当

#### 1. **学习率太高**
```yaml
optimizer:
  lr: 2.0e-4  # 对125M模型偏高
```
- 125M参数的模型推荐学习率：1e-4 或更低
- 过高的学习率导致训练不稳定

#### 2. **学习率调度器配置错误**
```yaml
scheduler:
  warmup_steps: 750
  total_steps: 37500    # 配置为2个epoch

training:
  max_epochs: 1          # 但实际只训练1个epoch
```

**计算分析**：
- 数据量：1,200,000条
- Batch size：2 × 32 (gradient_accumulation) = 64
- 每epoch步数：1,200,000 ÷ 64 = 18,750步
- **实际训练：18,750步 (1 epoch)**
- **Scheduler预期：37,500步 (2 epochs)**
- **不匹配！**导致学习率调度异常

#### 3. **严重过拟合**
- 训练数据：只用了675万中的120万条(18%)
- 训练方式：单epoch训练这120万条
- 结果：**模型完全记住了训练数据，Loss→0**
- 后果：泛化能力极差，生成时只会重复pattern

#### 4. **缺少repetition_penalty**
- generate方法没有惩罚重复token的机制
- 当模型过拟合时，更容易陷入重复循环

---

## 已完成的修复

### 1. ✅ 添加repetition_penalty支持

**修改文件**：
- `src/models/gpt2/model.py` - generate方法
- `src/trainer/base_trainer.py` - 评估配置
- `scripts/run_sft_training.py` - 参数传递
- `configs/train/gpt2_sft_chinanews.yaml` - 配置

**新参数**：
```yaml
evaluation:
  repetition_penalty: 1.2  # >1.0惩罚重复，推荐1.2-1.5
```

### 2. ✅ 创建修复后的训练配置

**新配置文件**：`configs/train/gpt2_sft_chinanews_fixed.yaml`

**关键改进**：
```yaml
optimizer:
  lr: 1.0e-4           # 降低学习率

scheduler:
  warmup_steps: 1000   # 增加warmup
  total_steps: 50000   # 2 epochs × 25000 steps，与实际匹配

training:
  max_epochs: 2        # 增加到2个epoch
  eval_interval: 1000  # 更频繁评估

data:
  max_samples: 1600000 # 增加数据量

evaluation:
  repetition_penalty: 1.2
```

---

## 解决方案

### 方案1：使用修复配置重新训练（推荐）

```bash
# 方式1: 使用便捷脚本
./retrain_fixed.sh

# 方式2: 直接运行
python scripts/run_sft_training.py \
    --config configs/train/gpt2_sft_chinanews_fixed.yaml
```

**预期效果**：
- 训练更稳定
- Loss正常收敛到2-4左右
- 生成质量显著改善
- 重复现象减少

**训练时间**：
- 2 epochs，约50,000步
- 按当前速度估计：~18小时

---

### 方案2：从当前checkpoint继续训练（不推荐）

虽然可以降低学习率继续训练，但不推荐，因为：
1. 模型已经严重过拟合
2. 很难恢复到正常状态
3. 重新训练效果更好

---

## 训练监控建议

### 关键指标

1. **Loss监控**
   - 初始Loss应该在6-8
   - 健康训练Loss应稳定在2-4
   - **如果Loss<1或趋近0，说明过拟合**

2. **生成质量**
   - 每1000步检查生成样本
   - 观察是否有重复pattern
   - 观察是否有连贯性

3. **学习率**
   - Warmup阶段应该逐渐上升
   - 之后应该平滑下降
   - 不应该有突变

### 异常信号

⚠️ **立即停止训练的信号**：
- Loss变成NaN或Inf
- Loss突然跳到很大的值
- 生成全是乱码或重复
- PPL突然爆炸（>1000）

⚠️ **需要调整的信号**：
- Loss下降太慢（>2000步还在6以上）
- Loss震荡剧烈
- 生成质量没有改善

---

## 调试工具

### 1. 数据处理调试
```bash
python debug_data_pipeline.py
```
检查：数据加载、tokenization、labels设置

### 2. 生成效果测试
```bash
python test_repetition_fix.py
```
测试不同repetition_penalty的效果

### 3. 监控训练
```bash
# 实时查看日志
tail -f checkpoints/gpt2_sft_chinanews_fixed/training_*.log

# 过滤关键信息
tail -f checkpoints/gpt2_sft_chinanews_fixed/training_*.log | grep -E "Loss:|生成样本"
```

---

## 进阶优化建议

### 1. 数据增强
- 使用更多数据（当前只用了18%）
- 数据去重
- 数据质量过滤

### 2. 训练策略
- 使用learning rate finder寻找最优学习率
- 尝试不同的warmup策略
- 考虑使用validation set监控过拟合

### 3. 模型架构
- 尝试增加dropout（当前attn_dropout=0.0）
- 使用weight decay防止过拟合
- 考虑layer normalization的位置

### 4. 生成策略
- 调整temperature (0.7-1.0)
- 调整top_p (0.85-0.95)
- 调整repetition_penalty (1.2-1.5)
- 尝试beam search

---

## 总结

### 问题根源
1. **学习率太高** (2e-4 → 1e-4)
2. **Scheduler配置错误** (total_steps不匹配)
3. **严重过拟合** (Loss→0)
4. **缺少repetition_penalty**

### 已修复
✅ 添加repetition_penalty支持  
✅ 创建修复后的训练配置  
✅ 降低学习率并修正scheduler  
✅ 增加数据量和训练epoch  
✅ 提供便捷重训练脚本

### 下一步
1. 运行 `./retrain_fixed.sh` 重新训练
2. 密切监控Loss和生成质量
3. 根据效果调整hyperparameters

---

## 附录：配置对比

| 参数 | 原配置 | 修复配置 | 说明 |
|------|--------|----------|------|
| 学习率 | 2e-4 | 1e-4 | 降低，更稳定 |
| Warmup steps | 750 | 1000 | 增加，更平滑 |
| Total steps | 37500 | 50000 | 匹配实际训练 |
| Max epochs | 1 | 2 | 增加训练充分性 |
| Max samples | 1.2M | 1.6M | 增加数据多样性 |
| Eval interval | 2000 | 1000 | 更频繁监控 |
| Repetition penalty | 无 | 1.2 | 防止重复生成 |

---

生成时间：2025-10-15  
诊断工具版本：v1.0
