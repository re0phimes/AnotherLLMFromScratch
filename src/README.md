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
