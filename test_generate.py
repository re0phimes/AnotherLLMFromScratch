"""
测试生成功能
"""
import torch
from transformers import AutoTokenizer
from src.models.gpt2.model import GPT2Model

def test_generate():
    """测试 GPT2Model 的 generate 方法"""
    print("=" * 70)
    print("测试 GPT2Model.generate() 方法")
    print("=" * 70)
    
    # 创建一个小模型用于测试
    model = GPT2Model(
        vocab_size=151936,  # Qwen2 词表大小
        n_layer=4,          # 减少层数以便快速测试
        n_head=4,
        n_embd=256,
        block_size=512,
        attn_dropout=0.0,
        resid_dropout=0.1,
    )
    
    # 使用 CPU 测试
    device = 'cpu'
    model = model.to(device)
    model.eval()
    
    # 加载 tokenizer
    print("\n加载 Qwen2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        print("✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        print("请确保已安装 transformers 库并有网络连接")
        return
    
    # 测试不同的生成策略
    test_prompts = [
        "中国经济正处于",
        "在未来的科技创新领域",
    ]
    
    print("\n" + "=" * 70)
    print("测试 1: 贪婪解码 (temperature=0)")
    print("=" * 70)
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        print(f"\nPrompt: {prompt}")
        print(f"Input IDs shape: {input_ids.shape}")
        
        try:
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.0,  # 贪婪解码
            )
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            print("✓ 生成成功")
        except Exception as e:
            print(f"✗ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试 2: Top-k 采样 (temperature=1.0, top_k=50)")
    print("=" * 70)
    
    prompt = test_prompts[0]
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(f"\nPrompt: {prompt}")
    
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=1.0,
            top_k=50,
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("✓ 生成成功")
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试 3: Top-p 采样 (temperature=0.8, top_p=0.9)")
    print("=" * 70)
    
    prompt = test_prompts[1]
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    print(f"\nPrompt: {prompt}")
    
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=20,
            temperature=0.8,
            top_p=0.9,
        )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        print("✓ 生成成功")
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_generate()
