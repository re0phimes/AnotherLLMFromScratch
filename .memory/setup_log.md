# 项目搭建日志

## 2025-09-30 - 初始化项目

### 已完成的操作：

1. **创建了 pyproject.toml**
   - 定义了项目基本信息和元数据
   - 声明了核心依赖：
     - torch >= 2.0.0 (深度学习框架)
     - datasets >= 2.14.0 (数据处理)
     - transformers >= 4.30.0 (仅用于tokenizer)
     - pyyaml >= 6.0 (配置文件解析)
     - tensorboard >= 2.13.0 (训练监控)
     - safetensors >= 0.4.0 (模型保存)
   - 配置了开发工具依赖（pytest, black, isort, flake8）
   - 设置了Python版本要求：>=3.12

2. **创建了 .memory/ 目录**
   - 用于记录项目开发的每一步操作
   - 符合用户规则要求

3. **创建了项目目录结构**
   - src/{dataset, model, trainer, utils}
   - configs/{model, train}
   - data/, scripts/, outputs/

4. **创建了main.py入口脚本**
   - 支持命令行参数解析（--config, --resume）
   - 支持分布式训练环境检测
   - 支持三种任务类型：pretrain, sft, dpo
   - 配置了输出目录和日志系统

5. **实现了src/utils/模块**
   - logger.py: 日志记录工具，支持文件和控制台输出，分布式友好
   - distributed.py: 分布式训练工具，自动检测torchrun环境
   - misc.py: 杂项工具，包括随机种子设置、配置管理、参数统计等
   - README.md: 模块使用文档

6. **实现了src/model/模块** ✅
   - **attention.py**: 
     * CausalSelfAttention - 标准的多头因果自注意力
     * FlashAttention - 使用PyTorch 2.0优化版本
     * 支持因果掩码、dropout、多头机制
   - **layer.py**:
     * MLP - 两层前馈网络，GELU激活
     * TransformerBlock - 完整的Transformer层，采用Pre-Norm结构
     * 残差连接和层归一化
   - **gpt.py**:
     * GPTModel - 完整的GPT模型
     * Token嵌入 + 位置嵌入
     * 多层Transformer堆叠
     * LM Head（与Token嵌入权重共享）
     * 支持训练模式（计算loss）和生成模式（generate方法）
     * 支持Temperature、Top-K、Top-P采样
   - **README.md**: 详细的模块文档和使用示例

### 当前进度总结：
✅ 已完成：
1. 项目基础架构（pyproject.toml）
2. 目录结构
3. main.py入口脚本
4. src/utils/ 工具模块
5. src/model/ 模型模块

🔄 进行中：
- 准备实现数据集和训练器模块

### 下一步计划：
- 实现src/dataset/模块（先实现SFT数据集）
- 实现src/trainer/模块（训练循环、优化器、检查点）
- 创建配置文件和调试数据集
- 运行第一次训练验证
