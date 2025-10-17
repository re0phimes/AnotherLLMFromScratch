"""
检查训练时实际的batch数据
看看是否有label=-100导致loss为0的问题
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoTokenizer
import yaml
from src.dataset.pretrain import PretrainDatasetModule
from src.models.modeling_auto import AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F


def check_batch_and_loss():
    """检查实际训练批次和loss计算"""
    
    print("="*80)
    print("检查训练批次和Loss计算")
    print("="*80)
    
    # 加载配置
    config_path = "configs/train/gpt2_sft_chinanews_fixed.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载tokenizer
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name_or_path'])
    
    # 加载数据
    print("加载数据集...")
    dataset_module = PretrainDatasetModule.from_config(
        config['data'],
        tokenizer=tokenizer,
        seed=42
    )
    dataset = dataset_module.build_dataset()
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个batch
    print("\n获取训练批次...")
    samples = [dataset[i] for i in range(3)]
    batch = dataset_module.collate_fn(samples)
    
    print(f"\nBatch信息:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    
    # 统计labels
    labels = batch['labels']
    total_labels = labels.numel()
    valid_labels = (labels != -100).sum().item()
    masked_labels = (labels == -100).sum().item()
    
    print(f"\nLabels统计:")
    print(f"  总label数: {total_labels}")
    print(f"  有效label (!=100): {valid_labels} ({valid_labels/total_labels*100:.2f}%)")
    print(f"  masked label (=-100): {masked_labels} ({masked_labels/total_labels*100:.2f}%)")
    
    if valid_labels == 0:
        print("\n❌ 严重错误: 所有labels都是-100，无法计算loss!")
        return
    
    # 加载模型
    print("\n加载模型...")
    model_config_path = config['model']['model_config_path']
    with open(model_config_path, 'r') as f:
        model_dict = yaml.safe_load(f)
    auto_config = AutoConfig.from_dict(model_dict)
    model = AutoModelForCausalLM.from_config(auto_config)
    model.eval()
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        outputs = model(batch['input_ids'])
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
    
    print(f"Logits shape: {logits.shape}")
    
    # 计算loss（按照SFTTrainer的方式）
    print("\n计算loss...")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    print(f"shift_logits shape: {shift_logits.shape}")
    print(f"shift_labels shape: {shift_labels.shape}")
    
    # 统计shift后的有效label
    shift_total = shift_labels.numel()
    shift_valid = (shift_labels != -100).sum().item()
    
    print(f"\nShift后的labels统计:")
    print(f"  总label数: {shift_total}")
    print(f"  有效label: {shift_valid} ({shift_valid/shift_total*100:.2f}%)")
    
    if shift_valid == 0:
        print("\n❌ 严重错误: Shift后所有labels都是-100!")
        return
    
    # 计算loss
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    print(f"\nLoss值: {loss.item():.6f}")
    
    if loss.item() < 0.01:
        print("⚠️  警告: Loss异常低，可能存在问题!")
    elif loss.item() > 15:
        print("⚠️  警告: Loss异常高!")
    else:
        print("✓ Loss在合理范围内（对于随机初始化的模型）")
    
    # 检查logits分布
    print("\n检查logits分布:")
    print(f"  logits mean: {logits.mean().item():.6f}")
    print(f"  logits std: {logits.std().item():.6f}")
    print(f"  logits min: {logits.min().item():.6f}")
    print(f"  logits max: {logits.max().item():.6f}")
    
    # 检查是否有nan或inf
    if torch.isnan(logits).any():
        print("❌ Logits包含NaN!")
    if torch.isinf(logits).any():
        print("❌ Logits包含Inf!")
    
    # 检查最大概率token
    probs = F.softmax(shift_logits, dim=-1)
    max_probs, max_indices = probs.max(dim=-1)
    
    print(f"\n预测概率分析:")
    print(f"  最大概率均值: {max_probs.mean().item():.6f}")
    print(f"  最大概率最大值: {max_probs.max().item():.6f}")
    print(f"  最大概率最小值: {max_probs.min().item():.6f}")
    
    if max_probs.mean().item() > 0.9:
        print("⚠️  警告: 模型输出过于确定，可能过拟合!")
    
    print("\n" + "="*80)
    print("检查完成!")
    print("="*80)


if __name__ == "__main__":
    check_batch_and_loss()
