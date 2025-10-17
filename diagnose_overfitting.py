"""
诊断为什么会超快速过拟合
"""
import torch
import json
from collections import Counter
from transformers import AutoTokenizer
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def check_data_diversity():
    """检查数据多样性"""
    print("="*80)
    print("1. 检查数据多样性")
    print("="*80)
    
    data_path = "/home/modelenv/chentianxuan/projects/open_source_data_process/data/chinanews_pretrain.jsonl"
    
    # 采样检查
    num_samples = 10000
    texts = []
    text_hashes = []
    
    print(f"\n采样 {num_samples} 条数据...")
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            text = data.get('text', '')
            texts.append(text)
            text_hashes.append(hash(text))
    
    # 检查重复
    unique_hashes = len(set(text_hashes))
    duplicate_rate = (num_samples - unique_hashes) / num_samples
    
    print(f"\n重复率分析:")
    print(f"  总样本: {num_samples}")
    print(f"  唯一样本: {unique_hashes}")
    print(f"  重复率: {duplicate_rate*100:.2f}%")
    
    if duplicate_rate > 0.1:
        print(f"  ⚠️  警告: 重复率过高!")
    else:
        print(f"  ✓ 重复率正常")
    
    # 检查长度分布
    lengths = [len(text) for text in texts]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    
    print(f"\n长度分布:")
    print(f"  平均: {avg_len:.0f} 字符")
    print(f"  最短: {min_len} 字符")
    print(f"  最长: {max_len} 字符")
    
    # 检查常见开头
    print(f"\n常见开头词（前20个字符）:")
    starts = [text[:20] for text in texts if len(text) >= 20]
    start_counter = Counter(starts)
    
    for start, count in start_counter.most_common(10):
        if count > 1:
            print(f"  '{start}...': {count} 次 ({count/num_samples*100:.2f}%)")
    
    print("\n" + "="*80)


def check_tokenizer_vocab():
    """检查tokenizer配置"""
    print("\n" + "="*80)
    print("2. 检查Tokenizer词表大小")
    print("="*80)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    
    print(f"\nTokenizer信息:")
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  模型最大长度: {tokenizer.model_max_length}")
    
    # 检查特殊token
    print(f"\n特殊tokens:")
    print(f"  pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"  eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"  bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    
    # 测试tokenization
    test_texts = [
        "中国经济正处于",
        "在未来的科技创新领域",
        "这是一个测试句子",
    ]
    
    print(f"\nTokenization测试:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        print(f"  '{text}': {len(tokens)} tokens")
    
    print("\n" + "="*80)


def check_model_config():
    """检查模型配置"""
    print("\n" + "="*80)
    print("3. 检查模型配置")
    print("="*80)
    
    model_config_path = "configs/model/gpt_125m.yaml"
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    print(f"\n模型配置:")
    print(f"  vocab_size: {model_config['vocab_size']}")
    print(f"  n_layer: {model_config['n_layer']}")
    print(f"  n_head: {model_config['n_head']}")
    print(f"  n_embd: {model_config['n_embd']}")
    print(f"  block_size: {model_config['block_size']}")
    print(f"  attn_dropout: {model_config['attn_dropout']}")
    print(f"  resid_dropout: {model_config['resid_dropout']}")
    
    # 计算参数量
    vocab_size = model_config['vocab_size']
    n_layer = model_config['n_layer']
    n_embd = model_config['n_embd']
    
    # Embedding层
    embedding_params = vocab_size * n_embd
    
    # Transformer层（简化计算）
    # 每层：4个权重矩阵（QKV + out）+ 2个MLP层
    transformer_params = n_layer * (
        4 * n_embd * n_embd +  # Attention
        2 * n_embd * (n_embd * model_config.get('mlp_multiplier', 4))  # MLP
    )
    
    # LM head (权重共享，不重复计算)
    total_params = embedding_params + transformer_params
    
    print(f"\n参数量估算:")
    print(f"  Embedding: {embedding_params:,}")
    print(f"  Transformer: {transformer_params:,}")
    print(f"  总计: {total_params:,}")
    
    # 检查dropout
    if model_config['attn_dropout'] == 0.0:
        print(f"\n⚠️  警告: attn_dropout=0，没有正则化，容易过拟合!")
    
    if model_config['resid_dropout'] < 0.1:
        print(f"⚠️  警告: resid_dropout={model_config['resid_dropout']}，正则化不足!")
    
    print("\n" + "="*80)


def check_training_config():
    """检查训练配置"""
    print("\n" + "="*80)
    print("4. 检查训练配置")
    print("="*80)
    
    config_path = "configs/train/gpt2_sft_chinanews_fixed.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n优化器配置:")
    print(f"  lr: {config['optimizer']['lr']}")
    print(f"  weight_decay: {config['optimizer']['weight_decay']}")
    print(f"  betas: {config['optimizer']['betas']}")
    
    print(f"\n调度器配置:")
    print(f"  warmup_steps: {config['scheduler']['warmup_steps']}")
    print(f"  total_steps: {config['scheduler']['total_steps']}")
    print(f"  min_lr: {config['scheduler']['min_lr']}")
    
    print(f"\n训练配置:")
    print(f"  max_epochs: {config['training']['max_epochs']}")
    print(f"  batch_size: {config['training']['micro_batch_size']}")
    print(f"  gradient_accumulation: {config['training']['gradient_accumulation']}")
    print(f"  max_grad_norm: {config['training']['max_grad_norm']}")
    print(f"  use_amp: {config['training']['use_amp']}")
    
    print(f"\n数据配置:")
    print(f"  max_samples: {config['data']['max_samples']}")
    print(f"  sequence_length: {config['data']['sequence_length']}")
    print(f"  shuffle: {config['data']['shuffle']}")
    
    # 计算有效batch size和数据覆盖
    effective_batch = config['training']['micro_batch_size'] * config['training']['gradient_accumulation']
    steps_per_epoch = config['data']['max_samples'] // effective_batch
    
    print(f"\n计算:")
    print(f"  有效batch size: {effective_batch}")
    print(f"  每epoch步数: {steps_per_epoch}")
    print(f"  总步数: {steps_per_epoch * config['training']['max_epochs']}")
    
    # 检查是否会快速过拟合
    samples_per_step = effective_batch
    unique_samples_per_6k_steps = samples_per_step * 6000
    
    print(f"\n过拟合风险分析:")
    print(f"  6000步处理的样本数: {unique_samples_per_6k_steps:,}")
    print(f"  数据集大小: {config['data']['max_samples']:,}")
    print(f"  数据覆盖率: {unique_samples_per_6k_steps / config['data']['max_samples'] * 100:.1f}%")
    
    if unique_samples_per_6k_steps >= config['data']['max_samples']:
        print(f"  ⚠️  6000步已经看完全部数据，可能过拟合!")
    else:
        print(f"  ✓ 6000步只看了部分数据")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    print("="*80)
    print("诊断快速过拟合问题")
    print("="*80)
    
    check_data_diversity()
    check_tokenizer_vocab()
    check_model_config()
    check_training_config()
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    
    print("\n总结：可能导致快速过拟合的原因:")
    print("1. 数据重复率过高")
    print("2. 模型dropout=0，缺乏正则化")
    print("3. 学习率仍然偏高")
    print("4. 数据相对简单，模型容量过大")
    print("\n请查看上面的输出找出具体原因。")


if __name__ == "__main__":
    main()
