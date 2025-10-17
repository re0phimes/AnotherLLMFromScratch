"""测试数据检查功能的脚本"""

import yaml
from pathlib import Path
from transformers import AutoTokenizer

from src.dataset.pretrain import PretrainDatasetModule
from src.utils import inspect_first_batch


def main():
    # 使用测试配置
    config_path = Path("configs/train/gpt2_pretrain_packed_test.yaml")
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    print("加载配置...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 加载 tokenizer
    tokenizer_path = config["model"]["tokenizer_name_or_path"]
    print(f"加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 构建数据集
    print("构建数据集...")
    data_cfg = config["data"]
    dataset_module = PretrainDatasetModule.from_config(
        data_cfg,
        tokenizer=tokenizer,
        seed=42,
    )
    
    # 构建 DataLoader
    print("构建 DataLoader...")
    dataloader = dataset_module.build_dataloader(
        batch_size=2,  # 小批次用于测试
        shuffle=False,
    )
    
    # 检查第一个 batch
    print("\n" + "=" * 100)
    print("开始数据检查...")
    print("=" * 100 + "\n")
    
    inspect_first_batch(dataloader, tokenizer, num_samples=2)
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
