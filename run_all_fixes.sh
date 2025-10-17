#!/bin/bash
#
# 完整的训练流程脚本
# 包括数据清洗（可选）和训练
#

set -e

echo "=================================="
echo "AnotherLLM训练修复脚本"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目目录
PROJECT_DIR="/home/modelenv/chentianxuan/projects/llm/AnotherLLMFromScratch"
DATA_DIR="/home/modelenv/chentianxuan/projects/open_source_data_process/data"

cd "$PROJECT_DIR"

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

echo ""
echo "=================================="
echo "步骤1: 数据质量检查"
echo "=================================="
echo ""

# 快速分析数据
echo "分析原始数据质量..."
python clean_data.py --mode analyze --input "$DATA_DIR/chinanews_pretrain.jsonl"

echo ""
read -p "是否需要清洗数据？(y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=================================="
    echo "步骤2: 清洗数据"
    echo "=================================="
    echo ""
    
    CLEANED_DATA="$DATA_DIR/chinanews_pretrain_cleaned.jsonl"
    
    echo "开始清洗数据..."
    python clean_data.py \
        --mode clean \
        --input "$DATA_DIR/chinanews_pretrain.jsonl" \
        --output "$CLEANED_DATA" \
        --max-duplicate 3 \
        --start-length 50 \
        --min-length 100
    
    echo ""
    echo -e "${GREEN}✓ 数据清洗完成${NC}"
    echo ""
    
    # 更新配置文件
    echo "更新训练配置..."
    sed -i "s|path: .*chinanews_pretrain.*|path: \"$CLEANED_DATA\"|g" \
        configs/train/gpt2_sft_chinanews_fixed.yaml
    
    echo -e "${GREEN}✓ 配置已更新为使用清洗后的数据${NC}"
else
    echo ""
    echo -e "${YELLOW}跳过数据清洗，使用原始数据${NC}"
fi

echo ""
echo "=================================="
echo "步骤3: 验证配置"
echo "=================================="
echo ""

# 显示当前配置
echo "当前配置:"
echo ""
echo "模型配置 (configs/model/gpt_125m.yaml):"
grep -E "vocab_size|attn_dropout|resid_dropout" configs/model/gpt_125m.yaml
echo ""
echo "训练配置 (configs/train/gpt2_sft_chinanews_fixed.yaml):"
grep -E "lr:|weight_decay:" configs/train/gpt2_sft_chinanews_fixed.yaml | head -2
grep -E "path:" configs/train/gpt2_sft_chinanews_fixed.yaml | grep data -A 1
echo ""

read -p "配置正确，继续训练？(y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练取消"
    exit 0
fi

echo ""
echo "=================================="
echo "步骤4: 清理旧checkpoint"
echo "=================================="
echo ""

if [ -d "checkpoints/gpt2_sft_chinanews_fixed" ]; then
    echo "发现旧的checkpoint目录"
    read -p "是否删除旧checkpoint从头开始？(y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧checkpoint..."
        rm -rf checkpoints/gpt2_sft_chinanews_fixed/*.pt
        echo -e "${GREEN}✓ 旧checkpoint已删除${NC}"
    else
        echo -e "${YELLOW}保留旧checkpoint，可能继续训练${NC}"
    fi
fi

echo ""
echo "=================================="
echo "步骤5: 开始训练"
echo "=================================="
echo ""

# 创建日志文件名
LOG_FILE="training_fixed_$(date +%Y%m%d_%H%M%S).log"

echo "训练配置:"
echo "  配置文件: configs/train/gpt2_sft_chinanews_fixed.yaml"
echo "  日志文件: $LOG_FILE"
echo "  预计时间: ~22-24小时"
echo ""

echo -e "${GREEN}开始训练...${NC}"
echo ""

# 运行训练
python scripts/run_sft_training.py \
    --config configs/train/gpt2_sft_chinanews_fixed.yaml \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=================================="
echo "训练完成！"
echo "=================================="
echo ""
echo "检查点保存在: checkpoints/gpt2_sft_chinanews_fixed/"
echo "训练日志: $LOG_FILE"
