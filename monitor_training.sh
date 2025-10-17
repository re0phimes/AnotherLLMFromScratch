#!/bin/bash
#
# 监控训练进度脚本
# 实时显示Loss和生成样本
#

LOG_DIR="checkpoints/gpt2_sft_chinanews_fixed/logs"
TRAINING_LOG=""

# 查找最新的日志文件
if [ -d "$LOG_DIR" ]; then
    TRAINING_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
fi

# 如果没有找到，查找当前目录的日志
if [ -z "$TRAINING_LOG" ] || [ ! -f "$TRAINING_LOG" ]; then
    TRAINING_LOG=$(ls -t training_*.log 2>/dev/null | head -1)
fi

if [ -z "$TRAINING_LOG" ] || [ ! -f "$TRAINING_LOG" ]; then
    echo "❌ 未找到训练日志文件"
    echo ""
    echo "可能的位置:"
    echo "  1. checkpoints/gpt2_sft_chinanews_fixed/logs/*.log"
    echo "  2. ./training_*.log"
    exit 1
fi

echo "=================================="
echo "训练监控"
echo "=================================="
echo "监控文件: $TRAINING_LOG"
echo ""
echo "提示: 按Ctrl+C退出监控（不会停止训练）"
echo ""

# 选择监控模式
echo "选择监控模式:"
echo "  1) 只看Loss"
echo "  2) 只看生成样本"
echo "  3) Loss + 生成样本"
echo "  4) 完整输出"
read -p "选择 (1-4): " -n 1 -r
echo
echo ""

case $REPLY in
    1)
        echo "监控Loss..."
        tail -f "$TRAINING_LOG" | grep --line-buffered "Loss:"
        ;;
    2)
        echo "监控生成样本..."
        tail -f "$TRAINING_LOG" | grep --line-buffered -A 2 "Generated"
        ;;
    3)
        echo "监控Loss和生成样本..."
        tail -f "$TRAINING_LOG" | grep --line-buffered -E "Loss:|Generated"
        ;;
    4)
        echo "显示完整输出..."
        tail -f "$TRAINING_LOG"
        ;;
    *)
        echo "无效选择，默认监控Loss"
        tail -f "$TRAINING_LOG" | grep --line-buffered "Loss:"
        ;;
esac
