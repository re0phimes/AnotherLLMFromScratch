"""
带生成评估的训练示例

演示如何在训练过程中定期生成文本样本来评估模型质量
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.modeling_auto import AutoModelForCausalLM, AutoConfig
from src.trainer.sft_trainer import SFTTrainer
from src.trainer.optimizer import configure_optimizer
from src.dataset.pretrain import PretrainDatasetModule


def main():
    """主训练流程"""
    # 加载配置
    config_path = project_root / "configs/train/gpt2_sft_chinanews.yaml"
    print(f"加载配置: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载 tokenizer
    tokenizer_name = config['model']['tokenizer_name_or_path']
    print(f"\n加载 tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # 加载模型配置
    model_config_path = config['model']['model_config_path']
    print(f"加载模型配置: {model_config_path}")
    
    with open(model_config_path, 'r') as f:
        model_config_dict = yaml.safe_load(f)
    
    # 创建模型
    model_config = AutoConfig.from_dict(model_config_dict)
    model = AutoModelForCausalLM.from_config(model_config)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建数据集
    print(f"\n创建数据集...")
    dataset_module = PretrainDatasetModule.from_config(
        config['data'],
        tokenizer=tokenizer
    )
    
    train_loader = dataset_module.build_dataloader(
        batch_size=config['training']['micro_batch_size'],
        shuffle=config['data'].get('shuffle', True),
        num_workers=config['data'].get('num_workers', 0),
        prefetch_factor=config['data'].get('prefetch_factor', None),
    )
    
    print(f"训练数据: {len(train_loader)} batches")
    
    # 配置优化器
    print(f"\n配置优化器...")
    optimizer = configure_optimizer(
        model=model,
        optimizer_name=config['optimizer']['name'],
        learning_rate=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=tuple(config['optimizer']['betas']),
        eps=config['optimizer']['eps'],
    )
    
    # 从配置中提取评估参数
    eval_config = config.get('evaluation', {})
    eval_prompts = eval_config.get('prompts', [])
    eval_max_tokens = eval_config.get('max_new_tokens', 100)
    
    # 创建训练器（带生成评估）
    print(f"\n创建训练器...")
    trainer = SFTTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=None,
        scheduler=None,  # 简化示例，不使用调度器
        max_epochs=config['training']['max_epochs'],
        grad_accum_steps=config['training']['gradient_accumulation'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_amp=config['training']['use_amp'],
        log_interval=config['training']['log_interval'],
        save_dir=config['training']['save_dir'],
        # 生成评估配置
        eval_interval=config['training'].get('eval_interval', 100),
        eval_prompts=eval_prompts,
        eval_max_tokens=eval_max_tokens,
        eval_temperature=0.8,
        eval_top_p=0.9,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    print(f"\n" + "=" * 70)
    print("开始训练（每 {eval_interval} 步生成一次样本）".format(
        eval_interval=config['training'].get('eval_interval', 100)
    ))
    print("=" * 70)
    
    trainer.train()


if __name__ == "__main__":
    main()
