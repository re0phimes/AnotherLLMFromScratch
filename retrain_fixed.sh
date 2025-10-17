#!/bin/bash
#
# 使用修复后的配置重新训练模型
#
# 主要改进：
# - 降低学习率到1e-4
# - 修正scheduler配置
# - 增加数据量和训练epoch
# - 添加repetition_penalty支持

set -e

echo "=========================================="
echo "开始使用修复配置训练模型"
echo "=========================================="

# 配置文件
CONFIG="configs/train/gpt2_sft_chinanews_fixed.yaml"

echo ""
echo "配置文件: $CONFIG"
echo ""

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 激活虚拟环境（如果需要）
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 显示配置
echo "训练配置:"
echo "  学习率: 1e-4 (降低，更稳定)"
echo "  Warmup steps: 1000"
echo "  Total steps: 50000 (2 epochs)"
echo "  数据量: 160万条"
echo "  Eval间隔: 1000步"
echo "  Repetition penalty: 1.2"
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消训练"
    exit 0
fi

# 开始训练
echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

python scripts/run_sft_training.py \
    --config "$CONFIG" \
    2>&1 | tee "checkpoints/gpt2_sft_chinanews_fixed/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "检查点保存在: checkpoints/gpt2_sft_chinanews_fixed/"
echo "训练日志已保存"
