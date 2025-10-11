# dataset 子系统说明

该目录负责“文本样本 ➝ 模型输入张量”的最终转换，供 Trainer 在训练循环中直接
调用 `model(**batch)`。所有模块遵循以下共性：

- **输入来源**：
  - 本地 JSONL：由数据准备脚本导出的轻量样本文件，每行一个 JSON，字段结构通过
    配置中的 `data`/`data_sources` 指定。
  - Streaming 数据：使用 Hugging Face `datasets` 等库远程迭代，配置信息同样来自
    `configs/train/*.yaml`。
- **输出格式**：统一返回字典形式的张量批次（`input_ids`、`attention_mask`、
  任务特定的 `labels` 或多路输入），Trainer 无需再做分词或手动 padding。
- **tokenizer 传入方式**：训练入口脚本构造 tokenizer，实例化 DatasetModule 时通过
  `from_config(..., tokenizer=...)` 注入，确保所有批次共享一致的特殊 token、padding
  策略。

## 模块一览

### `base.py`
- 定义 `DataSourceConfig` / `DataConfig` 以及 `parse_data_config`，解析 YAML 中的 `data`
  块（支持单源或 `data_sources` 混合）。
- 提供 `BaseDatasetModule` 抽象类，封装 `from_config`、`build_dataloader`、常用
  num_workers/prefetch 配置；子类只需实现 `build_dataset` 与 `collate_fn`。

### `pretrain.py`
- 面向自回归预训练：读取纯文本样本（JSONL 或 streaming），在 `collate_fn` 中完成
  tokenizer 批处理和 padding。
- 输出：`input_ids`、`attention_mask`、`labels`（padding 位置为 -100）、`metadata`。
- 文件顶部给出具体 tensor 示例，便于理解 mask 与 padding 行为。

### `sft.py`
- 面向指令微调：支持常见的 `instruction`/`input`/`output`（可通过配置覆盖），将
  prompt 与 response 拼接后编码。
- `labels` 中 prompt 部分被置为 -100，确保 loss 仅作用于响应段；`metadata` 记录
  prompt token 长度便于日志分析。

### `dpo.py`
- 面向偏好比较任务：样本包含 `prompt`、`chosen`、`rejected` 三段文本，分别编码
  并返回长度信息，方便 Trainer 直接计算偏好差值。

## 与配置文件的交互
- 训练配置 (`configs/train/*.yaml`) 中的 `data` 块支持：
  - `type: "local"` + `path`: 指向 JSONL 文件，可附带 `sampling_weight`、`max_samples`。
  - `type: "streaming"` + `provider`/`path`/`subset`: 描述远程数据集来源与附加参数。
  - `data_sources`: 数组形式，可混合多个来源；权重与采样逻辑由基类统一处理。
- DatasetModule 解析这些配置并构建底层 Dataset / IterableDataset，Trainer 只需调用
  `module.build_dataloader(batch_size=...)` 即可获取可训练的 batch。

## 扩展建议
- 新任务（如 RLHF 在线阶段）可继承 `BaseDatasetModule` 并添加对应文件，保持与现有
  模块一致的入参/输出约定。
- 若新增数据字段，请在模块顶部注释与本 README 中同步说明输入结构，便于团队协作。
