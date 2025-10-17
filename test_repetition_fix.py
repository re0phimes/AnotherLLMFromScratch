"""
测试repetition_penalty修复效果
使用已训练的checkpoint测试生成，对比有无repetition_penalty的效果
"""
import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.models.gpt2.model import GPT2Model
import yaml


def load_model_from_checkpoint(checkpoint_path, model_config_path):
    """从checkpoint加载模型"""
    # 加载模型配置
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # 创建模型
    model = GPT2Model(
        vocab_size=model_config['vocab_size'],
        n_layer=model_config['n_layer'],
        n_head=model_config['n_head'],
        n_embd=model_config['n_embd'],
        block_size=model_config['block_size'],
        attn_dropout=model_config.get('attn_dropout', 0.0),
        resid_dropout=model_config.get('resid_dropout', 0.1),
        mlp_multiplier=model_config.get('mlp_multiplier', 4.0),
        activation=model_config.get('activation', 'gelu'),
        layer_norm_eps=model_config.get('layer_norm_eps', 1e-5),
        qkv_bias=model_config.get('qkv_bias', True),
        use_flash=model_config.get('use_flash', True),
        pad_token_id=model_config.get('pad_token_id', None),
    )
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def test_generation(model, tokenizer, prompts, device='cpu'):
    """测试生成效果"""
    model = model.to(device)
    model.eval()
    
    print("\n" + "="*80)
    print("测试生成效果")
    print("="*80)
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print("="*80)
        
        # 测试1: 无repetition_penalty (原始问题)
        print("\n[测试1] 无repetition_penalty:")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=80,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.0,  # 不惩罚
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"生成: {generated_text}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        # 测试2: 使用repetition_penalty=1.2 (轻度惩罚)
        print("\n[测试2] repetition_penalty=1.2:")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=80,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"生成: {generated_text}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        # 测试3: 使用repetition_penalty=1.5 (中度惩罚)
        print("\n[测试3] repetition_penalty=1.5:")
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=80,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.5,
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"生成: {generated_text}")
        except Exception as e:
            print(f"生成失败: {e}")
        
        print("\n" + "-"*80)


def main():
    """主函数"""
    print("="*80)
    print("测试repetition_penalty修复效果")
    print("="*80)
    
    # 配置路径
    checkpoint_path = "checkpoints/gpt2_sft_chinanews/checkpoint_epoch_0.pt"
    model_config_path = "configs/model/gpt_125m.yaml"
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        print(f"错误: Checkpoint不存在: {checkpoint_path}")
        return
    
    if not Path(model_config_path).exists():
        print(f"错误: 模型配置不存在: {model_config_path}")
        return
    
    # 加载tokenizer
    print("\n加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    print("✓ Tokenizer加载成功")
    
    # 加载模型
    print("\n加载模型...")
    model = load_model_from_checkpoint(checkpoint_path, model_config_path)
    print("✓ 模型加载成功")
    
    # 测试prompts
    test_prompts = [
        "中国经济正处于",
        "在未来的科技创新领域",
        "china is one the age of",
    ]
    
    # 使用CPU测试 (如果有GPU可以改为'cuda')
    device = 'cpu'
    
    # 运行测试
    test_generation(model, tokenizer, test_prompts, device)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print("\n建议:")
    print("1. 如果repetition_penalty=1.2效果不够好，可以尝试1.3-1.5")
    print("2. 如果生成质量仍然很差，说明模型训练本身有问题，需要重新训练")
    print("3. 可以尝试调整temperature (0.7-1.0) 和 top_p (0.85-0.95)")
    print("="*80)


if __name__ == "__main__":
    main()
