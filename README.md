# AnotherLLMFromScratch
从0构建一个分布式大模型训练


   自动检查（推荐）
```
    # 训练前自动检查数据
    python scripts/run_sft_training.py \
        --config configs/train/gpt2_pretrain_packed.yaml

独立测试

    # 只检查数据，不训练
    python test_data_inspection.py

禁用检查

    # 恢复训练时跳过检查
    python scripts/run_sft_training.py \
        --config configs/train/gpt2_pretrain_packed.yaml \
        --no-inspect-data
```