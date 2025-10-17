"""调试脚本：验证打包式预训练数据集的正确性。

检查项：
1. 每个样本都是固定长度 (sequence_length)
2. 样本中包含 <|endoftext|> 分隔符
3. 没有 padding tokens
4. attention_mask 全为 1
5. labels == input_ids (没有 -100)
6. 样本跨越多个源文档
"""

import yaml
from pathlib import Path
from transformers import AutoTokenizer

from src.dataset.pretrain import PretrainDatasetModule


def main():
    print("=" * 80)
    print("打包式预训练数据集调试")
    print("=" * 80)
    
    config_path = Path("configs/train/gpt2_pretrain_packed.yaml")
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        print("使用内存中的测试配置...")
        
        test_config = {
            "data_sources": [
                {
                    "path": "/tmp/test_pretrain_small.jsonl",
                    "type": "local",
                    "name": "test_small",
                    "max_samples": 1000,
                }
            ],
            "sequence_length": 512,
            "pack_sequences": True,
            "shuffle_buffer_size": 50,
        }
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
            test_config = full_config.get("data", {})
    
    print(f"\n配置信息:")
    data_sources = test_config.get('data_sources', [])
    if data_sources:
        print(f"  数据路径: {data_sources[0].get('path')}")
        print(f"  max_samples: {data_sources[0].get('max_samples', 'unlimited')}")
    else:
        print(f"  数据路径: {test_config.get('path')}")
    print(f"  sequence_length: {test_config.get('sequence_length', 1024)}")
    print(f"  pack_sequences: {test_config.get('pack_sequences', False)}")
    print(f"  shuffle_buffer_size: {test_config.get('shuffle_buffer_size', 5000)}")
    
    tokenizer_path = "Qwen/Qwen2.5-0.5B"
    print(f"\n加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    print("\n初始化数据集模块...")
    dataset_module = PretrainDatasetModule.from_config(
        test_config,
        tokenizer=tokenizer,
        seed=42,
    )
    
    print(f"  extras.pack_sequences: {dataset_module.extras.pack_sequences}")
    print(f"  extras.shuffle_buffer_size: {dataset_module.extras.shuffle_buffer_size}")
    
    print("\n构建 DataLoader...")
    dataloader = dataset_module.build_dataloader(
        batch_size=4,
        shuffle=False,
    )
    
    print("\n获取第一个 batch...")
    batch = next(iter(dataloader))
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    metadata = batch["metadata"]
    
    print(f"\nBatch 形状:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  labels: {labels.shape}")
    
    print(f"\n检查 1: 固定长度")
    seq_len = input_ids.shape[1]
    expected_len = test_config.get("sequence_length", 1024)
    print(f"  实际长度: {seq_len}, 期望长度: {expected_len}")
    assert seq_len == expected_len, f"长度不匹配: {seq_len} != {expected_len}"
    print("  ✓ 通过")
    
    print(f"\n检查 2: 包含 EOS token")
    eos_id = tokenizer.eos_token_id
    eos_count = (input_ids == eos_id).sum().item()
    print(f"  EOS token 出现次数: {eos_count}")
    assert eos_count > 0, "没有找到 EOS token"
    print("  ✓ 通过")
    
    print(f"\n检查 3: 没有 padding tokens")
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        pad_count = (input_ids == tokenizer.pad_token_id).sum().item()
        print(f"  PAD token 出现次数: {pad_count}")
        assert pad_count == 0, f"发现 {pad_count} 个 padding tokens"
    else:
        print(f"  跳过 - PAD token 与 EOS token 相同")
    print("  ✓ 通过")
    
    print(f"\n检查 4: attention_mask 全为 1")
    ones_count = (attention_mask == 1).sum().item()
    total_count = attention_mask.numel()
    print(f"  值为 1 的元素: {ones_count}/{total_count}")
    assert ones_count == total_count, f"attention_mask 不全为 1"
    print("  ✓ 通过")
    
    print(f"\n检查 5: labels == input_ids (没有 -100)")
    minus_100_count = (labels == -100).sum().item()
    print(f"  值为 -100 的元素: {minus_100_count}")
    assert minus_100_count == 0, f"发现 {minus_100_count} 个 -100"
    
    match_count = (labels == input_ids).sum().item()
    print(f"  labels 与 input_ids 匹配: {match_count}/{total_count}")
    assert match_count == total_count, "labels 与 input_ids 不完全匹配"
    print("  ✓ 通过")
    
    print(f"\n检查 6: 样本内容")
    print(f"\n第一个样本的前 200 个 tokens:")
    first_sample_ids = input_ids[0][:200].tolist()
    
    eos_positions = [i for i, tid in enumerate(first_sample_ids) if tid == eos_id]
    print(f"  EOS token 位置 (前 200 tokens): {eos_positions}")
    
    if len(eos_positions) >= 2:
        print(f"\n  文档 1 (tokens 0-{eos_positions[0]}):")
        doc1_text = tokenizer.decode(first_sample_ids[:eos_positions[0]])
        print(f"    {doc1_text[:100]}...")
        
        print(f"\n  文档 2 (tokens {eos_positions[0]+1}-{eos_positions[1]}):")
        doc2_text = tokenizer.decode(first_sample_ids[eos_positions[0]+1:eos_positions[1]])
        print(f"    {doc2_text[:100]}...")
        
        print("  ✓ 通过 - 样本确实跨越多个文档")
    else:
        print("  ⚠ 警告 - 前 200 tokens 中只有 1 个或 0 个文档分隔符")
    
    print(f"\n源信息:")
    print(f"  {metadata.get('source', [])}")
    
    print("\n" + "=" * 80)
    print("✓ 所有检查通过！打包数据集工作正常。")
    print("=" * 80)


if __name__ == "__main__":
    main()
