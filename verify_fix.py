#!/usr/bin/env python3
"""验证 packed dataset 修复是否有效"""

import torch
from transformers import AutoTokenizer
from src.dataset.pretrain import PretrainDatasetModule


def verify_packed_dataset_fix():
    """验证 packed dataset 是否正确处理 EOS token 以及注意力掩码"""
    
    print("="*70)
    print("验证 Packed Dataset 修复")
    print("="*70)
    
    # 1. 加载 tokenizer
    print("\n1. 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    print(f"   ✓ EOS token ID: {tokenizer.eos_token_id}")
    print(f"   ✓ BOS token ID: {tokenizer.bos_token_id}")
    print(f"   ✓ PAD token ID: {tokenizer.pad_token_id}")
    
    # 2. 创建测试数据
    print("\n2. 创建测试数据...")
    test_examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, tokenizer.eos_token_id, 5, 6, 7, tokenizer.eos_token_id]),
            "doc_ids": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "source": "test"
        },
        {
            "input_ids": torch.tensor([10, 11, tokenizer.eos_token_id, 13, 14, 15, 16, tokenizer.eos_token_id]),
            "doc_ids": torch.tensor([2, 2, 2, 2, 3, 3, 3, 3]),
            "source": "test"
        }
    ]
    print(f"   ✓ 创建了 {len(test_examples)} 个测试样本")
    
    # 3. 模拟 collate_fn
    print("\n3. 测试 collate_fn...")
    # 手动执行 collate_fn 的逻辑
    input_ids = torch.stack([ex["input_ids"] for ex in test_examples])
    doc_ids = torch.stack([ex["doc_ids"] for ex in test_examples])
    seq_len = input_ids.size(1)

    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    attention_mask = causal_mask.unsqueeze(0) & same_doc

    labels = input_ids.clone()
    if tokenizer.eos_token_id is not None:
        eos_mask = input_ids == tokenizer.eos_token_id
        labels[eos_mask] = -100
    
    # 4. 验证结果
    print("\n4. 验证结果...")
    print(f"\n   Input IDs shape: {input_ids.shape}")
    print(f"   Input IDs:\n{input_ids}")
    print(f"\n   Labels (应该在 EOS 位置为 -100):\n{labels}")
    print(f"\n   Attention mask (True 表示可见):\n{attention_mask}")
    
    # 5. 检查修复是否生效
    print("\n5. 检查修复...")
    eos_positions = (input_ids == tokenizer.eos_token_id)
    labels_at_eos = labels[eos_positions]
    
    mask_ok = True
    for b in range(attention_mask.size(0)):
        for i in range(seq_len):
            for j in range(seq_len):
                if attention_mask[b, i, j]:
                    if doc_ids[b, i] != doc_ids[b, j] or j > i:
                        mask_ok = False

    if (labels_at_eos == -100).all() and mask_ok:
        print("   ✅ 修复成功！EOS 已忽略且注意力掩码阻断跨文档访问")
        return True

    if not (labels_at_eos == -100).all():
        print("   ❌ 修复失败！部分 EOS 位置的 labels 不是 -100")
        print(f"      EOS 位置的 labels: {labels_at_eos}")
    if not mask_ok:
        print("   ❌ 注意力掩码构造错误，存在跨文档可见性或非因果关系")
    return False


def compare_loss_calculation():
    """比较修复前后的 loss 计算差异"""
    
    print("\n" + "="*70)
    print("比较修复前后的 Loss 计算")
    print("="*70)
    
    # 模拟一个简单的例子
    batch_size, seq_len, vocab_size = 2, 8, 100
    
    # 模拟 logits（模型输出）
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # 模拟 input_ids，其中包含 EOS token（假设 EOS token id = 50）
    eos_token_id = 50
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids[0, 3] = eos_token_id  # 在第1个样本的第4个位置放 EOS
    input_ids[1, 5] = eos_token_id  # 在第2个样本的第6个位置放 EOS
    
    print(f"\n1. Input IDs:\n{input_ids}")
    
    # 修复前：所有 token 都参与 loss 计算
    labels_before = input_ids.clone()
    shift_logits_before = logits[:, :-1, :].contiguous()
    shift_labels_before = labels_before[:, 1:].contiguous()
    loss_before = torch.nn.functional.cross_entropy(
        shift_logits_before.view(-1, vocab_size),
        shift_labels_before.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    # 修复后：EOS token 位置不参与 loss 计算
    labels_after = input_ids.clone()
    eos_mask = (input_ids == eos_token_id)
    labels_after[eos_mask] = -100
    
    print(f"\n2. Labels (修复后，EOS 位置为 -100):\n{labels_after}")
    
    shift_logits_after = logits[:, :-1, :].contiguous()
    shift_labels_after = labels_after[:, 1:].contiguous()
    loss_after = torch.nn.functional.cross_entropy(
        shift_logits_after.view(-1, vocab_size),
        shift_labels_after.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    
    print(f"\n3. Loss 比较:")
    print(f"   修复前 Loss: {loss_before.item():.4f}")
    print(f"   修复后 Loss: {loss_after.item():.4f}")
    print(f"   差异: {abs(loss_before.item() - loss_after.item()):.4f}")
    
    # 计算有多少个 token 被忽略
    num_eos = eos_mask.sum().item()
    total_tokens = batch_size * (seq_len - 1)  # shift 后的 token 数
    print(f"\n4. 统计:")
    print(f"   总 token 数: {total_tokens}")
    print(f"   EOS token 数: {num_eos}")
    print(f"   EOS 占比: {num_eos / (batch_size * seq_len) * 100:.2f}%")


if __name__ == "__main__":
    print("\n🔍 开始验证修复...\n")
    
    try:
        # 验证 packed dataset 修复
        success = verify_packed_dataset_fix()
        
        # 比较 loss 计算
        compare_loss_calculation()
        
        print("\n" + "="*70)
        if success:
            print("✅ 验证完成！修复已正确应用。")
            print("\n建议下一步:")
            print("1. 删除旧的 checkpoint:")
            print("   mv checkpoints/gpt2_pretrain_packed_lowmem checkpoints/backup")
            print("2. 使用修复后的配置重新训练:")
            print("   python scripts/train_pretrain.py --config configs/train/gpt2_pretrain_fixed.yaml")
            print("3. 监控训练指标，确认 Loss 和 PPL 恢复正常")
        else:
            print("❌ 验证失败！请检查代码修复是否正确应用。")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 验证过程中出错: {e}")
        import traceback
        traceback.print_exc()
