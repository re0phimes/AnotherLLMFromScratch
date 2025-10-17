"""
测试训练一个step，看看loss是否正常
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.dataset.pretrain import PretrainDatasetModule
from src.models.modeling_auto import AutoConfig, AutoModelForCausalLM
from src.trainer.optimizer import configure_optimizer


def test_one_training_step():
    """测试一个完整的训练step"""
    
    print("="*80)
    print("测试训练一个step")
    print("="*80)
    
    # 加载配置
    config_path = "configs/train/gpt2_sft_chinanews_fixed.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载tokenizer
    print("\n1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name_or_path'])
    
    # 加载数据
    print("2. 加载数据集...")
    dataset_module = PretrainDatasetModule.from_config(
        config['data'],
        tokenizer=tokenizer,
        seed=42
    )
    dataset = dataset_module.build_dataset()
    print(f"   数据集大小: {len(dataset)}")
    
    # 获取一个batch
    print("3. 获取训练batch...")
    samples = [dataset[i] for i in range(2)]
    batch = dataset_module.collate_fn(samples)
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"   Batch shape: {input_ids.shape}")
    
    # 统计有效labels
    valid_labels = (labels != -100).sum().item()
    print(f"   有效labels: {valid_labels} / {labels.numel()} ({valid_labels/labels.numel()*100:.1f}%)")
    
    # 加载模型
    print("\n4. 加载模型...")
    model_config_path = config['model']['model_config_path']
    with open(model_config_path, 'r') as f:
        model_dict = yaml.safe_load(f)
    auto_config = AutoConfig.from_dict(model_dict)
    model = AutoModelForCausalLM.from_config(auto_config)
    model = model.to(device)
    
    # 检查模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")
    
    # 检查权重初始化
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"   {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            break  # 只看第一个
    
    # 配置优化器
    print("\n5. 配置优化器...")
    optimizer = configure_optimizer(
        model,
        optimizer_type=config['optimizer']['name'],
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=tuple(config['optimizer']['betas']),
        eps=config['optimizer']['eps']
    )
    print(f"   学习率: {config['optimizer']['lr']}")
    
    # 前向传播（初始状态）
    print("\n6. 前向传播（训练前）...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 计算初始loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        initial_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        print(f"   初始Loss: {initial_loss.item():.6f}")
        
        if initial_loss.item() < 1.0:
            print("   ⚠️  警告: 初始Loss异常低!")
        elif initial_loss.item() > 15.0:
            print("   ⚠️  警告: 初始Loss异常高!")
        else:
            print("   ✓ 初始Loss正常")
    
    # 训练一步
    print("\n7. 训练一个step...")
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(input_ids)
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    
    # 计算loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    print(f"   Loss: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print("\n8. 检查梯度...")
    total_grad_norm = 0.0
    has_nan = False
    has_zero = True
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            
            if torch.isnan(param.grad).any():
                print(f"   ❌ {name}: 梯度包含NaN!")
                has_nan = True
            
            if param.grad.abs().max().item() > 0:
                has_zero = False
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"   总梯度范数: {total_grad_norm:.6f}")
    
    if has_nan:
        print("   ❌ 检测到NaN梯度!")
    elif has_zero:
        print("   ⚠️  所有梯度都是0!")
    else:
        print("   ✓ 梯度正常")
    
    # 梯度裁剪
    max_grad_norm = config['training']['max_grad_norm']
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    print(f"   梯度裁剪阈值: {max_grad_norm}")
    
    # 优化器更新
    optimizer.step()
    print("   ✓ 参数更新完成")
    
    # 检查更新后的loss
    print("\n9. 检查更新后的loss...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        after_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        print(f"   更新后Loss: {after_loss.item():.6f}")
        loss_change = initial_loss.item() - after_loss.item()
        print(f"   Loss变化: {loss_change:+.6f}")
        
        if abs(loss_change) < 1e-6:
            print("   ⚠️  Loss几乎没有变化!")
        elif loss_change > 0:
            print("   ✓ Loss下降（正常）")
        else:
            print("   ⚠️  Loss上升!")
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    
    print("\n总结:")
    print(f"  初始Loss: {initial_loss.item():.6f}")
    print(f"  训练Loss: {loss.item():.6f}")
    print(f"  更新后Loss: {after_loss.item():.6f}")
    print(f"  梯度范数: {total_grad_norm:.6f}")
    
    if initial_loss.item() < 1.0:
        print("\n⚠️  问题: 初始Loss异常低，模型初始化可能有问题")
    elif total_grad_norm < 1e-6:
        print("\n⚠️  问题: 梯度太小，模型无法学习")
    elif abs(loss_change) < 1e-6:
        print("\n⚠️  问题: Loss不下降，优化器可能有问题")
    else:
        print("\n✓ 训练流程看起来正常")


if __name__ == "__main__":
    test_one_training_step()
