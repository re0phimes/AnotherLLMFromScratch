# 项目架构
```
AnotherLLMFromScratch/
├── configs/
│   ├── model/
│   │   ├── gpt_125m.yaml
│   │   └── gpt_1b.yaml
│   └── train/
│       ├── pretrain.yaml
│       ├── sft.yaml
│       └── dpo.yaml
│   └── README.md
│
├── data/
│   ├── openwebtext_subset.jsonl
│   └── alpaca_data.json
│   └── README.md
│
├── scripts/
│   ├── run_pretrain.sh
│   └── run_sft.sh
│   └── README.md
│
├── src/
│   ├── dataset/
│   │   ├── base.py
│   │   ├── pretrain.py
│   │   ├── sft.py
│   │   └── dpo.py
│   │   └── README.md
│   │
│   ├── models/
│   │   ├── components/
│   │   │   ├── attention/
│   │   │   │   ├── multi_head.py
│   │   │   │   └── grouped_query.py
│   │   │   ├── embeddings/
│   │   │   │   ├── learned.py
│   │   │   │   └── rotary.py
│   │   │   ├── norms.py
│   │   │   ├── mlp.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── gpt2/
│   │   │   ├── attention.py
│   │   │   ├── layer.py
│   │   │   ├── model.py
│   │   │   └── config.py
│   │   │
│   │   ├── qwen2/
│   │   │   ├── attention.py
│   │   │   ├── layer.py
│   │   │   ├── model.py
│   │   │   └── config.py
│   │   │
│   │   ├── modeling_auto.py
│   │   └── __init__.py
│   │
│   ├── trainer/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── checkpoint.py
│   │   └── README.md
│   │
│   ├── utils/
│   │   ├── distributed.py
│   │   └── logger.py
│   │   └── README.md
│   │
│   ├── train_pretrain.py
│   ├── train_sft.py
│   └── train_dpo.py
│   └── README.md
│
├── requirements.txt
└── README.md
```

## 3.2 模块与文件详细规格
### 3.2.1 configs/ - 配置模块
configs/README.md:
要求: 必须说明此目录是框架的“控制大脑”，解释模型配置与训练配置分离的设计哲学。提供一个如何创建新的 .yaml 配置文件的模板和指南。
configs/model/*.yaml:
要求: 定义模型物理结构。必须包含 n_layer, n_head, n_embd, vocab_size, block_size。
configs/train/*.yaml:
要求: 定义一次完整的训练任务。必须包含 model_config_path 和一个结构化的 data 块。该 data 块需支持本地和流式两种模式，例如：data: { type: "local", path: "..." } 或 data: { type: "streaming", provider: "huggingface", path: "...", subset: "..." }。
### 3.2.2 data/ - 原始数据存储
data/README.md:
要求: 必须在文件顶部用醒目方式声明：此目录仅用于存放小规模的开发和调试样本数据，绝不可用于存放大规模训练数据。必须指引用户去 configs/train/ 目录下配置大规模数据的来源。
### 3.2.3 scripts/ - 运行脚本
scripts/README.md:
要求: 必须提供每个脚本的用途说明和使用示例。例如，如何通过修改 NUM_GPUS 变量来控制单卡或多卡训练。
### 3.2.4 src/ - 核心源代码
src/README.md:
要求: 提供 src 目录下各个子模块（dataset, model, trainer, utils）的顶层设计思想和职责划分，帮助开发者快速建立代码心智地图。
### 3.2.5 src/dataset/ - 数据处理子系统
dataset/README.md:
要求: 必须解释数据处理模块的设计理念。明确指出此模块负责处理离线（Offline）数据集。对于需要在线数据生成的强化学习任务（如PPO），需说明此模块仅负责提供初始的提示（Prompt）数据源，完整的经验生成逻辑不在此模块范围内。
pretrain.py, sft.py, dpo.py:
要求: 必须能够解析 configs/train/*.yaml 中结构化的 data 配置块，并根据 type 字段决定是加载本地文件还是通过相应库（如 datasets）进行流式读取。
### 3.2.6 src/models/ - 模型定义子系统
models 目录采用“组件库 + 架构实现”双层组织，以满足多模型族的快速组合需求。

models/components/README.md:
要求: 阐述组件化设计理念，说明注意力、归一化、MLP、嵌入等基础模块可以跨模型族复用，并给出如何在新架构中选型这些组件的指南。

components/attention/
- multi_head.py: 实现 GPT-2 风格的标准多头注意力，支持因果掩码、Flash Attention 自动回退。
- grouped_query.py: 实现 GQA/MQA 注意力，集成 RoPE、KV Cache 写入逻辑，供 Qwen2 等现代模型使用。

components/embeddings/
- learned.py: 可学习的位置编码实现，用于 GPT-2 类模型。
- rotary.py: RoPE 旋转位置编码实现，包含缓存与 apply 接口，可在生成模式下复用。

components/norms.py:
提供 LayerNorm、RMSNorm 等规范化层。

components/mlp.py:
提供标准 GELU MLP、SwiGLU 等前馈网络实现，支持配置宽度系数与激活函数。

models/gpt2/
- attention.py: 基于 components.attention.multi_head.Wrapper，实现 GPT-2 专用注意力模块（含 Dropout、因果掩码控制）。
- layer.py: 构建 GPT-2 Transformer Block（Pre-Norm + LayerNorm + GELU MLP + Residual）。
- model.py: 堆叠 Block，提供完整 GPT-2 like 模型（含嵌入、语言建模头、generate 接口）。
- config.py: 使用 dataclass/pydantic 定义 GPT-2 配置，含 vocab_size、n_layer、n_head、n_embd、max_position 等属性。

models/qwen2/
- attention.py: 基于 grouped_query 注意力与 RoPE，支持 KV Cache、动态 position_ids。
- layer.py: 构建 Qwen2 Transformer Block（RMSNorm、SwiGLU、残差结构、dropout 策略）。
- model.py: 完整 Qwen2 like 模型，实现高效生成路径、KV Cache 维护。
- config.py: 定义 Qwen2 配置，涵盖 num_kv_heads、ffn_multiplier、rope_base、rope_scaling、use_sliding_window 等参数。

modeling_auto.py:
实现自动模型工厂，例如 AutoModelForCausalLM，根据配置中的 model_family 自动导入 gpt2 或 qwen2 的 model.py。

models/__init__.py:
导出常用模型类、配置类与 Auto 工厂入口。
### 3.2.7 src/trainer/ - 训练驱动子系统
trainer/README.md:
要求: 必须说明 trainer.py 是驱动所有训练任务的通用引擎，它负责执行训练循环的通用部分（如梯度累积、AMP、DDP封装），而将任务特定的逻辑（如损失计算）委托给模型本身。
### 3.2.8 src/utils/ - 公共工具箱
utils/README.md:
要求: 简要说明 distributed.py, logger.py, misc.py 各自提供的核心功能，方便其他模块开发者查找和复用。
### 3.2.9 src/train_*.py - 任务入口脚本
要求: 每个脚本的职责仅限于：解析配置、实例化组件（模型、数据集）、实例化 Trainer 并启动训练。
### 3.2.10 根目录文件
requirements.txt:
要求: 必须列出项目运行所需的所有 Python 依赖库及其精确版本号。
README.md:
要求: 作为项目的总入口文档，必须提供清晰的项目介绍、环境配置指南（如何安装 requirements.txt）、快速开始（如何使用 scripts/ 目录下的脚本运行一个SFT示例）、以及对整体架构的简要说明。