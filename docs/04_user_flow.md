# 4. 用户工作流 (User Workflow)

本文档为开发者提供了一个端到端的操作指南，详细描述了从环境设置到启动训练、再到自定义实验的完整工作流程。

---

### 4.1 环境设置与安装 (Environment Setup & Installation)

这是开始任何开发工作前的“第一步”，旨在创建一个干净、隔离且可复现的开发环境。

1.  **获取代码**:
    从版本控制系统克隆项目仓库到本地。
    ```bash
    git clone https://github.com/re0phimes/AnotherLLMFromScratch
    cd ANOTHERLLMFROMSCRATCH
    ```

2.  **创建虚拟环境**:
    我们使用 `uv` 来创建一个与系统 Python 环境隔离的虚拟环境。此命令将在项目根目录下创建一个名为 `.venv` 的文件夹。我们明确指定使用 Python 3.12。
    ```bash
    # 确保你的系统中安装了 Python 3.12，并且 uv 可以找到它
    uv venv -p 3.12
    ```

3.  **激活虚拟环境**:
    激活后，所有 Python 相关的命令（如 `python`, `pip`）都将在此隔离环境中执行。
    ```bash
    # 对于 macOS / Linux (bash/zsh)
    source .venv/bin/activate

    # 对于 Windows (PowerShell)
    # .venv\Scripts\Activate.ps1
    ```
    激活成功后，你的终端提示符前通常会显示 `(.venv)`。

4.  **安装项目依赖**:
    此命令会读取 `pyproject.toml` 文件，并使用 `uv` 的高速安装器安装所有必需的依赖。
    ```bash
    uv pip install -e .
    ```
    > **说明**: `-e .` 参数代表“可编辑模式”（editable install）。它会将当前项目以链接的方式安装到虚拟环境中，这意味着你对 `src/` 目录下的源代码所做的任何修改，都会立即在环境中生效，无需重新安装。这对于开发和调试至关重要。

---

### 4.2 运行第一次训练 (Running the First Training)

项目的核心设计是“配置驱动”。你通过一个主脚本 (`main.py`) 和一个指定的配置文件来启动所有任务。

1.  **理解入口点**:
    项目的唯一执行入口是根目录下的 `main.py` 脚本。它负责解析命令行参数、加载配置、并根据配置初始化和运行训练流程。

2.  **选择配置文件**:
    所有的实验配置都存放在 `configs/` 目录下。为了快速验证环境是否配置成功，我们提供一个用于调试的微型训练配置。

3.  **启动训练**:
    使用以下命令启动一个简单的“监督微调 (SFT)”任务。这个任务会使用一个极小的数据集和模型，在 CPU 或单块 GPU 上快速完成。
    ```bash
    python main.py --config configs/sft/debug_sft_on_tiny_story.yaml
    ```

---

### 4.3 查看训练输出 (Inspecting the Output)

为了保持工作区的整洁和实验的可追溯性，所有训练产生的文件都将被保存在一个结构化的输出目录中。

- **输出根目录**: `outputs/`
- **单次运行目录结构**: 每次运行时，系统会根据实验名和时间戳创建一个唯一的子目录，例如 `outputs/debug_sft/2025-09-30_10-00-00/`。

在该目录中，你会找到：
- `config.yaml`: 本次运行所使用的**完整配置文件的快照**，用于精确复现。
- `logs/`: 存放训练日志文件，例如 TensorBoard 日志，用于可视化监控损失、学习率等指标。
- `checkpoints/`: 存放训练过程中保存的模型检查点。每个检查点包含模型权重、优化器状态等，用于从中断处恢复训练（满足 `ER-3.2` 需求）。

---

### 4.4 自定义与实验 (Customization & Experimentation)

这是本项目的核心学习循环。我们强烈推荐遵循以下“复制-修改-运行”的模式来进行你自己的实验。

**黄金原则**: 永远不要直接修改 `configs/` 目录下的官方示例配置。

1.  **复制配置**:
    从一个最接近你目标的示例配置开始，复制它并赋予一个描述性的新名字。
    ```bash
    cp configs/sft/debug_sft_on_tiny_story.yaml configs/sft/my_first_experiment.yaml
    ```

2.  **修改配置**:
    用你喜欢的编辑器打开 `my_first_experiment.yaml`。你可以调整各种参数，例如：
    - 更改学习率: `optimizer.params.lr: 5.0e-5`
    - 增加模型层数: `model.n_layer: 8`
    - 更换数据集: `data.dataset_name: "HuggingFaceH4/ultrachat_200k"`

3.  **运行新实验**:
    使用新的配置文件启动训练。
    ```bash
    python main.py --config configs/sft/my_first_experiment.yaml
    ```
    一个新的输出目录将在 `outputs/my_first_experiment/...` 下被创建，与之前的调试运行完全隔离。

---

### 4.5 启动分布式训练 (Launching Distributed Training)

当需要在单台机器的多张 GPU 上进行训练时（满足 `FR-4.2` 需求），我们使用 PyTorch 的标准启动器 `torchrun`。

假设你希望在 4 张 GPU 上进行数据并行训练，命令如下：
```bash
torchrun --nproc_per_node=4 main.py --config configs/pretrain/1b_model_pretrain.yaml