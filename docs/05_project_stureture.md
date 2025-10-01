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
│   ├── model/
│   │   ├── attention.py
│   │   ├── layer.py
│   │   └── gpt.py
│   │   └── README.md
│   │
│   ├── trainer/
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── checkpoint.py
│   │   └── README.md
│   │
│   ├── utils/
│   │   ├── distributed.py
│   │   ├── logger.py
│   │   └── misc.py
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
### 3.2.6 src/model/ - 模型定义子系统
model/README.md:
要求: 必须解释模型构建的层次化设计思想：attention.py 是核心计算单元，layer.py 将其组装成可复用的Transformer Block，gpt.py 最终将Block堆叠成完整模型。
layer.py:
要求: 职责: 实现一个完整的、可复用的Transformer Block。规格: 该模块必须导入并使用 torch.nn.LayerNorm。它负责将 attention.py 提供的自注意力模块、一个前馈网络（MLP）、层归一化和残差连接，按照标准的Pre-Norm结构组装起来。
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