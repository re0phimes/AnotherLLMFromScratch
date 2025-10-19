# src 目录概览

本目录汇聚训练框架的核心代码，按照模块职责划分：

- `dataset/`：从离线/Streaming 文本样本生成模型可直接使用的张量批次。
- `models/`：模型组件与具体架构实现（GPT-2、Qwen2 等）。
- `trainer/`：训练循环、优化器、checkpoint 管理等通用训练逻辑。
- `utils/`：公共工具（日志、分布式、misc）支持。
- `train_*.py`：任务入口脚本，绑定配置文件，实例化模型/数据/Trainer。

## 数据流简述

1. **原始数据准备**：在仓库外部或 `scripts/` 中完成清洗与 JSONL 导出（或配置
   streaming 来源）。
2. **配置驱动**：在 `configs/train/*.yaml` 的 `data` 块中声明数据源、混合比例、
   序列长度与 tokenizer 等信息。
3. **DatasetModule（`dataset/`）**：
   - 读取配置并构建 Dataset。
   - 使用传入的 tokenizer 生成 batch：如 `PretrainDatasetModule` 输出
     `{"input_ids", "attention_mask", "labels", "metadata"}`，其中 labels 的 padding
     位置为 -100。
   - `SFTDatasetModule` 将 prompt/response 拼接后编码，prompt 部分在 labels 中置 -100。
   - `DPODatasetModule` 独立编码 prompt/chosen/rejected，返回带长度信息的张量。
4. **模型层（`models/`）**：如 `GPT2Model` 接收上述张量批次，按配置执行前向，返回
   logits/loss。
5. **Trainer（`trainer/`）**：负责迭代 dataloader、调用模型、反向传播、优化器更新、
   checkpoint 保存等。Trainer 与 DatasetModule 之间只通过字典批次交互，无需额外
   分词或 padding。
6. **任务入口脚本**：`train_pretrain.py` 等脚本读取 YAML，实例化 tokenizer、
   DatasetModule、模型、Trainer，并触发完整训练。

## 子模块说明

- `dataset/README.md`：详述输入 JSONL/Streaming 的字段约定、输出张量结构、各模块
  职责及扩展方式。
- `models/`：组件化设计，`components/` 提供注意力、MLP、norm 等基础模块，
  `/gpt2`、`/qwen2` 等文件夹实现具体架构，`modeling_auto.py` 提供工厂模式按配置加载。
- `trainer/`（待完善）：计划包含 `Trainer` 抽象、优化器封装、AMP/分布式支持、
  checkpoint 恢复等。
- `utils/`：日志、分布式初始化、通用工具函数，供其他模块复用。

通过以上分层，项目实现了“配置驱动、模块解耦”的训练框架：数据的输入输出语义
在 `dataset/` 层统一，模型可专注于前向逻辑，Trainer 聚焦在训练调度，入口脚本则
负责按配置组装整个流水线。

## 调用配置穿透

### 模型配置

训练用的就是你仓库里的自定义模型实现（不是HF现成模型本体，只有分词器用HF）。加载链路如下：
- scripts/run_sft_training.py → load_model() 读取 configs/model/gpt_125m.yaml
- src/models/modeling_auto.py → AutoConfig.from_dict() 解析为 GPT2Config → AutoModelForCausalLM.from_config()
- 注册表映射 "gpt2" → src/models/gpt2/model.py 的 GPT2Model（内部用你写的 LearnedAbsoluteEmbedding、MultiHeadSelfAttention 等组件）

因此最终实例化并训练的是你在 src/models/gpt2/model.py 定义的 GPT2Model。


### 其他训练配置穿透

配置“穿透”链路：max_steps 与 max_new_tokens

####max_steps

- YAML
```yaml
training:
max_epochs: 1
max_steps: 60000
```

- scripts/run_sft_training.py → prepare_components()
     ```python
     training_cfg = config.get("training", {})
     max_epochs = args.max_epochs or int(training_cfg.get("max_epochs", 1))
     max_steps = training_cfg.get("max_steps")              # ← 读取
     if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
         if max_steps is None:
             steps_per_epoch = 1000                         # ← 仅用于估算/日志
         else:
             steps_per_epoch = max_steps // max_epochs      # ← 估算每轮步数
     else:
         steps_per_epoch = len(train_loader)

     total_steps = max_epochs * steps_per_epoch             # ← 仅用于 scheduler
     scheduler = create_scheduler(optimizer, scheduler_cfg, total_steps)
     ```
- scripts/run_sft_training.py → run_training()（传给 Trainer 仅作显示/ETA）
     ```python
     trainer = SFTTrainer(..., estimated_steps_per_epoch=artifacts.steps_per_epoch, ...)
     ```
- src/trainer/base_trainer.py
     ```python
     # __init__
     self._estimated_steps_per_epoch = estimated_steps_per_epoch
     # _log_training_step() 里仅用于：
     batch_info = f"Step {current_step_in_epoch}/{self._estimated_steps_per_epoch} in epoch"
     ```

重要说明：当前代码“不会用 max_steps 做硬停止”。它只影响学习率调度 total_steps 和日志里的 “.../N in epoch、ETA”。若要真正跑到 N 步停，需要在 BaseTrainer.train_epoch() 中加上基于
      self.global_step 的 break 条件，或封装一个只产出 N×grad_accum 批次的数据迭代器。

#### max_new_tokens
YAML
     ```yaml
     evaluation:
       max_new_tokens: 1024
     ```
scripts/run_sft_training.py → run_training()
     ```python
     eval_cfg = config.get("evaluation", {})
     eval_max_tokens = eval_cfg.get("max_new_tokens", 100)  # ← 读取
     trainer = SFTTrainer(..., eval_max_tokens=eval_max_tokens, ...)
     ```
src/trainer/base_trainer.py
     ```python
     # __init__
     self.eval_max_tokens = kwargs.get('eval_max_tokens', 100)
     # generate_samples()
     generated_ids = model.generate(..., max_new_tokens=self.eval_max_tokens, ...)
     ```
src/models/gpt2/model.py
     ```python
     def generate(self, input_ids, max_new_tokens, ...): ...
     ```
说明：max_new_tokens 只影响评估阶段生成长度，不影响训练步数/损失收敛；但会增加每次评估的墙钟时间。