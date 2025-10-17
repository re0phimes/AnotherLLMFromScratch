"""
调试训练数据处理流程
检查数据预处理、tokenization和labels是否正确
"""
import torch
from transformers import AutoTokenizer
from pathlib import Path
import yaml
import sys
import json

sys.path.append(str(Path(__file__).parent))

from src.dataset.pretrain import PretrainDatasetModule


def check_raw_data(data_path, num_samples=5):
    """检查原始数据"""
    print("="*80)
    print("1. 检查原始数据")
    print("="*80)
    
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line)
            text = data.get('text', '')
            print(f"\n样本 {i+1}:")
            print(f"  长度: {len(text)} 字符")
            print(f"  前100字符: {text[:100]}")
            print(f"  后100字符: {text[-100:]}")
    print("\n" + "="*80)


def check_tokenizer(tokenizer):
    """检查tokenizer配置"""
    print("\n" + "="*80)
    print("2. 检查Tokenizer配置")
    print("="*80)
    
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    print(f"unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    
    # 测试tokenization
    test_text = "中国经济正处于"
    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"\n测试文本: '{test_text}'")
    print(f"Token IDs: {tokens}")
    print(f"Token数量: {len(tokens)}")
    decoded = tokenizer.decode(tokens)
    print(f"解码结果: '{decoded}'")
    print("\n" + "="*80)


def check_dataset_module(config_path, tokenizer, num_samples=3):
    """检查数据集模块处理"""
    print("\n" + "="*80)
    print("3. 检查数据集处理流程")
    print("="*80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    data_cfg = full_config['data']
    print(f"\n数据配置:")
    print(f"  path: {data_cfg.get('path')}")
    print(f"  type: {data_cfg.get('type')}")
    print(f"  sequence_length: {data_cfg.get('sequence_length')}")
    print(f"  max_samples: {data_cfg.get('max_samples')}")
    
    # 创建dataset模块
    print("\n创建PretrainDatasetModule...")
    dataset_module = PretrainDatasetModule.from_config(
        data_cfg,
        tokenizer=tokenizer,
        seed=42
    )
    print("✓ DatasetModule创建成功")
    
    # 创建dataset
    print("\n构建Dataset...")
    dataset = dataset_module.build_dataset()
    print(f"✓ Dataset大小: {len(dataset)}")
    
    # 检查原始样本
    print("\n检查原始样本 (collate前):")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"  keys: {sample.keys()}")
        print(f"  text长度: {len(sample.get('text', ''))} 字符")
        print(f"  text前50字符: {sample.get('text', '')[:50]}")
    
    # 检查collate后的batch
    print("\n" + "-"*80)
    print("检查collate_fn处理后的batch:")
    print("-"*80)
    
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    batch = dataset_module.collate_fn(samples)
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"\ninput_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    
    # 详细检查每个样本
    for i in range(batch['input_ids'].shape[0]):
        print(f"\n{'='*60}")
        print(f"样本 {i+1} 详细信息:")
        print(f"{'='*60}")
        
        input_ids = batch['input_ids'][i]
        attention_mask = batch['attention_mask'][i]
        labels = batch['labels'][i]
        
        # 统计信息
        total_tokens = input_ids.shape[0]
        valid_tokens = (attention_mask == 1).sum().item()
        pad_tokens = (attention_mask == 0).sum().item()
        label_mask_count = (labels == -100).sum().item()
        valid_labels = (labels != -100).sum().item()
        
        print(f"\n统计:")
        print(f"  总token数: {total_tokens}")
        print(f"  有效token数 (attention_mask==1): {valid_tokens}")
        print(f"  padding token数 (attention_mask==0): {pad_tokens}")
        print(f"  label=-100的数量: {label_mask_count}")
        print(f"  有效label数 (label!=-100): {valid_labels}")
        
        # 检查是否有效
        if valid_labels == 0:
            print("\n⚠️ 警告: 所有labels都是-100！这会导致loss=0")
        
        # 显示前10个和后10个token
        print(f"\n前10个tokens:")
        print(f"  input_ids: {input_ids[:10].tolist()}")
        print(f"  attention_mask: {attention_mask[:10].tolist()}")
        print(f"  labels: {labels[:10].tolist()}")
        
        print(f"\n后10个tokens:")
        print(f"  input_ids: {input_ids[-10:].tolist()}")
        print(f"  attention_mask: {attention_mask[-10:].tolist()}")
        print(f"  labels: {labels[-10:].tolist()}")
        
        # 解码查看内容
        valid_input_ids = input_ids[attention_mask == 1]
        decoded_text = tokenizer.decode(valid_input_ids, skip_special_tokens=False)
        print(f"\n解码内容 (前100字符):")
        print(f"  {decoded_text[:100]}")
    
    print("\n" + "="*80)
    
    return batch


def check_loss_computation(batch, vocab_size=151936):
    """模拟loss计算"""
    print("\n" + "="*80)
    print("4. 模拟Loss计算")
    print("="*80)
    
    # 创建随机logits模拟模型输出
    batch_size, seq_len = batch['input_ids'].shape
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = batch['labels']
    
    # 按照SFTTrainer的方式计算loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    print(f"shift_logits shape: {shift_logits.shape}")
    print(f"shift_labels shape: {shift_labels.shape}")
    
    # 统计有效label
    valid_label_count = (shift_labels != -100).sum().item()
    total_label_count = shift_labels.numel()
    
    print(f"\n有效label比例: {valid_label_count}/{total_label_count} = {valid_label_count/total_label_count*100:.2f}%")
    
    if valid_label_count == 0:
        print("\n❌ 严重错误: 没有任何有效的label用于计算loss!")
        print("这会导致loss=nan或0，并且模型无法学习")
        return None
    
    # 计算loss
    import torch.nn.functional as F
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    print(f"\n模拟loss值: {loss.item():.4f}")
    print("(注意: 这是随机logits的loss，仅用于验证计算流程)")
    
    print("\n" + "="*80)
    
    return loss


def main():
    """主函数"""
    print("="*80)
    print("调试训练数据处理流程")
    print("="*80)
    
    # 配置路径
    config_path = "configs/train/gpt2_sft_chinanews.yaml"
    
    # 加载配置获取数据路径
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    
    print(f"\n配置文件: {config_path}")
    print(f"数据文件: {data_path}")
    
    # 1. 检查原始数据
    check_raw_data(data_path, num_samples=3)
    
    # 2. 加载并检查tokenizer
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    print("✓ Tokenizer加载成功")
    check_tokenizer(tokenizer)
    
    # 3. 检查数据集处理
    batch = check_dataset_module(config_path, tokenizer, num_samples=3)
    
    # 4. 检查loss计算
    if batch is not None:
        check_loss_computation(batch)
    
    print("\n" + "="*80)
    print("调试完成!")
    print("="*80)
    
    print("\n总结:")
    print("请检查上面的输出，特别关注:")
    print("1. 是否有 '所有labels都是-100' 的警告")
    print("2. 有效label的比例是否合理 (应该>0%)")
    print("3. tokenization是否正常")
    print("4. 数据内容是否符合预期")
    print("="*80)


if __name__ == "__main__":
    main()
