# file: src/utils/data_inspection.py
# Description: 数据检查工具，用于在训练前验证数据处理流程
# 显示原始数据、tokenize 后的结果，确保数据处理正确

from typing import Any, Dict, Optional
import torch


def inspect_batch(
    batch: Dict[str, Any],
    tokenizer: Any,
    max_samples: int = 2,
    max_seq_display: int = 50,
) -> None:
    """检查并打印 batch 的详细信息。
    
    Args:
        batch: 包含 input_ids, attention_mask, labels 的 batch
        tokenizer: 用于解码 token IDs 的 tokenizer
        max_samples: 最多显示几个样本
        max_seq_display: 每个样本最多显示多少个 tokens
    """
    print("\n" + "=" * 100)
    print("📋 BATCH 数据检查")
    print("=" * 100)
    
    # 1. Batch 结构信息
    print("\n1️⃣  Batch 结构:")
    print(f"   Keys: {list(batch.keys())}")
    
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    labels = batch.get("labels")
    metadata = batch.get("metadata", {})
    
    if input_ids is None:
        print("   ❌ 错误: batch 中没有 input_ids")
        return
    
    # 2. 形状信息
    print("\n2️⃣  张量形状:")
    print(f"   input_ids shape:      {input_ids.shape}")
    if attention_mask is not None:
        print(f"   attention_mask shape: {attention_mask.shape}")
    if labels is not None:
        print(f"   labels shape:         {labels.shape}")
    
    batch_size, seq_len = input_ids.shape
    print(f"\n   📊 Batch size: {batch_size}")
    print(f"   📏 Sequence length: {seq_len}")
    
    # 3. 统计信息
    print("\n3️⃣  Token 统计:")
    
    # 特殊 token IDs
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id
    
    print(f"   Tokenizer 特殊 tokens:")
    print(f"      PAD token ID: {pad_token_id}")
    print(f"      EOS token ID: {eos_token_id}")
    print(f"      BOS token ID: {bos_token_id}")
    
    # 统计特殊 token 出现次数
    total_tokens = input_ids.numel()
    if pad_token_id is not None:
        pad_count = (input_ids == pad_token_id).sum().item()
        print(f"\n   PAD tokens: {pad_count}/{total_tokens} ({pad_count/total_tokens*100:.2f}%)")
    
    if eos_token_id is not None:
        eos_count = (input_ids == eos_token_id).sum().item()
        print(f"   EOS tokens: {eos_count}/{total_tokens} ({eos_count/total_tokens*100:.2f}%)")
    
    if bos_token_id is not None:
        bos_count = (input_ids == bos_token_id).sum().item()
        print(f"   BOS tokens: {bos_count}/{total_tokens} ({bos_count/total_tokens*100:.2f}%)")
    
    # 4. Attention mask 检查
    if attention_mask is not None:
        print("\n4️⃣  Attention Mask:")
        ones_count = (attention_mask == 1).sum().item()
        zeros_count = (attention_mask == 0).sum().item()
        total = attention_mask.numel()
        print(f"   值为 1: {ones_count}/{total} ({ones_count/total*100:.2f}%)")
        print(f"   值为 0: {zeros_count}/{total} ({zeros_count/total*100:.2f}%)")
    
    # 5. Labels 检查
    if labels is not None:
        print("\n5️⃣  Labels:")
        ignore_count = (labels == -100).sum().item()
        valid_count = (labels != -100).sum().item()
        total = labels.numel()
        print(f"   有效 labels: {valid_count}/{total} ({valid_count/total*100:.2f}%)")
        print(f"   忽略 labels (-100): {ignore_count}/{total} ({ignore_count/total*100:.2f}%)")
        
        # 检查 labels 是否等于 input_ids
        if valid_count > 0:
            labels_match = (labels[labels != -100] == input_ids[labels != -100]).all().item()
            if labels_match:
                print(f"   ✅ 有效 labels 与 input_ids 完全匹配")
            else:
                print(f"   ⚠️  有效 labels 与 input_ids 不完全匹配")
    
    # 6. Metadata
    if metadata:
        print("\n6️⃣  Metadata:")
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)) and len(value) > 0:
                print(f"   {key}: {value[:min(3, len(value))]}{'...' if len(value) > 3 else ''}")
            else:
                print(f"   {key}: {value}")
    
    # 7. 样本详细信息
    print("\n" + "=" * 100)
    print(f"7️⃣  样本详细内容 (显示前 {min(max_samples, batch_size)} 个样本)")
    print("=" * 100)
    
    for idx in range(min(max_samples, batch_size)):
        print(f"\n📄 样本 #{idx + 1}:")
        print("-" * 100)
        
        sample_input_ids = input_ids[idx]
        sample_attention_mask = attention_mask[idx] if attention_mask is not None else None
        sample_labels = labels[idx] if labels is not None else None
        
        # 首先显示完整的样本统计信息
        print(f"\n   📏 样本完整长度: {seq_len} tokens")
        
        # 统计这个样本中的特殊 tokens
        sample_stats = []
        if eos_token_id is not None:
            eos_in_sample = (sample_input_ids == eos_token_id).sum().item()
            sample_stats.append(f"EOS: {eos_in_sample}")
        if pad_token_id is not None and pad_token_id != eos_token_id:
            pad_in_sample = (sample_input_ids == pad_token_id).sum().item()
            sample_stats.append(f"PAD: {pad_in_sample}")
        if bos_token_id is not None:
            bos_in_sample = (sample_input_ids == bos_token_id).sum().item()
            sample_stats.append(f"BOS: {bos_in_sample}")
        
        if sample_stats:
            print(f"   🔖 特殊 tokens: {', '.join(sample_stats)}")
        
        # A. 显示前 N 个 token IDs
        display_len = min(max_seq_display, seq_len)
        print(f"\n   A. Token IDs (前 {display_len}/{seq_len} 个):")
        token_ids_list = sample_input_ids[:display_len].tolist()
        print(f"      {token_ids_list}")
        
        # B. 显示对应的 attention mask
        if sample_attention_mask is not None:
            print(f"\n   B. Attention Mask (前 {display_len} 个):")
            mask_list = sample_attention_mask[:display_len].tolist()
            print(f"      {mask_list}")
        
        # C. 显示对应的 labels
        if sample_labels is not None:
            print(f"\n   C. Labels (前 {display_len} 个):")
            labels_list = sample_labels[:display_len].tolist()
            print(f"      {labels_list}")
        
        # D. 找到 EOS token 位置（完整列表）
        if eos_token_id is not None:
            eos_positions = (sample_input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_pos_list = eos_positions.tolist()
                print(f"\n   D. EOS Token 位置 (共 {len(eos_pos_list)} 个):")
                
                # 如果 EOS 位置很多，分行显示
                if len(eos_pos_list) <= 20:
                    print(f"      {eos_pos_list}")
                else:
                    # 显示前10个和后10个
                    print(f"      前10个: {eos_pos_list[:10]}")
                    print(f"      ... (省略 {len(eos_pos_list) - 20} 个)")
                    print(f"      后10个: {eos_pos_list[-10:]}")
                
                # 计算每个文档段落的长度
                print(f"\n   📊 文档段落长度分布:")
                doc_lengths = []
                prev_pos = 0
                for pos in eos_pos_list:
                    doc_len = pos - prev_pos
                    doc_lengths.append(doc_len)
                    prev_pos = pos + 1  # 下一个文档从 EOS 后开始
                
                # 如果还有剩余 tokens（最后一个 EOS 之后）
                if prev_pos < seq_len:
                    last_doc_len = seq_len - prev_pos
                    doc_lengths.append(last_doc_len)
                
                # 显示统计信息
                if doc_lengths:
                    print(f"      文档数量: {len(doc_lengths)}")
                    print(f"      平均长度: {sum(doc_lengths) / len(doc_lengths):.1f} tokens")
                    print(f"      最短: {min(doc_lengths)} tokens")
                    print(f"      最长: {max(doc_lengths)} tokens")
                    
                    # 显示前几个文档的长度
                    if len(doc_lengths) <= 10:
                        print(f"      各文档长度: {doc_lengths}")
                    else:
                        print(f"      前5个文档长度: {doc_lengths[:5]}")
                        print(f"      后5个文档长度: {doc_lengths[-5:]}")
            else:
                print(f"\n   D. ⚠️  警告: 样本中没有找到 EOS token (ID: {eos_token_id})")
        else:
            print(f"\n   D. ℹ️  Tokenizer 没有定义 EOS token")
        
        # E. 解码文本（显示样本开头、中间、结尾）
        print(f"\n   E. 解码后的文本:")
        try:
            # 1. 显示开头部分
            head_len = min(100, seq_len // 3)
            head_text = tokenizer.decode(
                sample_input_ids[:head_len].tolist(),
                skip_special_tokens=False
            )
            print(f"\n      🔹 开头 ({head_len} tokens):")
            print(f"      {repr(head_text[:300])}{'...' if len(head_text) > 300 else ''}")
            
            # 2. 显示中间部分
            if seq_len > 200:
                mid_start = seq_len // 2 - 50
                mid_end = seq_len // 2 + 50
                mid_text = tokenizer.decode(
                    sample_input_ids[mid_start:mid_end].tolist(),
                    skip_special_tokens=False
                )
                print(f"\n      🔹 中间 (位置 {mid_start}-{mid_end}):")
                print(f"      {repr(mid_text[:300])}{'...' if len(mid_text) > 300 else ''}")
            
            # 3. 显示结尾部分
            tail_len = min(100, seq_len // 3)
            tail_text = tokenizer.decode(
                sample_input_ids[-tail_len:].tolist(),
                skip_special_tokens=False
            )
            print(f"\n      🔹 结尾 (最后 {tail_len} tokens):")
            print(f"      {repr(tail_text[:300])}{'...' if len(tail_text) > 300 else ''}")
            
        except Exception as e:
            print(f"      ❌ 解码失败: {e}")
        
        # F. 如果是打包数据，显示文档分隔的详细信息
        if eos_token_id is not None and len(eos_positions) > 0:
            print(f"\n   F. 文档拼接验证:")
            print(f"      {'='*90}")
            print(f"      ✅ 样本包含 {len(eos_positions)} 个 EOS 分隔符")
            print(f"      📚 文档数量: {len(doc_lengths)} 个片段")
            print(f"      {'='*90}")
            
            # 显示前 3 个文档的详细内容
            num_docs_to_show = min(3, len(doc_lengths))
            print(f"\n      📖 显示前 {num_docs_to_show} 个文档片段:")
            
            for doc_idx in range(num_docs_to_show):
                if doc_idx == 0:
                    start = 0
                    end = eos_positions[doc_idx].item()
                elif doc_idx < len(eos_positions):
                    start = eos_positions[doc_idx - 1].item() + 1
                    end = eos_positions[doc_idx].item()
                else:
                    start = eos_positions[doc_idx - 1].item() + 1
                    end = seq_len
                
                doc_len = end - start
                
                if doc_len > 0:
                    print(f"\n      ┌─ 文档片段 #{doc_idx + 1} ─────────────────────────")
                    print(f"      │ 位置: [{start}:{end}]")
                    print(f"      │ 长度: {doc_len} tokens")
                    
                    # 解码文档内容（不包含特殊 tokens）
                    doc_tokens = sample_input_ids[start:end].tolist()
                    doc_text = tokenizer.decode(doc_tokens, skip_special_tokens=True)
                    
                    # 显示文档内容（限制长度）
                    max_display_chars = 200
                    if len(doc_text) <= max_display_chars:
                        print(f"      │ 内容: {repr(doc_text)}")
                    else:
                        print(f"      │ 内容: {repr(doc_text[:max_display_chars])}...")
                        print(f"      │       (总共 {len(doc_text)} 字符)")
                    
                    print(f"      └{'─'*50}")
            
            # 如果有更多文档，提示一下
            if len(doc_lengths) > num_docs_to_show:
                remaining = len(doc_lengths) - num_docs_to_show
                print(f"\n      ... 还有 {remaining} 个文档片段（未显示）")
            
            # 验证拼接正确性
            print(f"\n      🔍 拼接正确性验证:")
            
            # 检查相邻 EOS 之间是否有内容
            has_empty_segments = any(length == 0 for length in doc_lengths)
            if has_empty_segments:
                print(f"      ⚠️  警告: 存在空文档片段（连续的 EOS tokens）")
            else:
                print(f"      ✅ 所有文档片段都有内容（无连续 EOS）")
            
            # 检查是否使用了正确的分隔符
            eos_token_str = tokenizer.decode([eos_token_id])
            print(f"      ✅ 分隔符 token: {repr(eos_token_str)} (ID: {eos_token_id})")
            
            # 显示完整拼接模式（简化）
            if len(doc_lengths) >= 2:
                print(f"\n      📋 拼接模式示意:")
                pattern_parts = []
                for i in range(min(3, len(doc_lengths))):
                    pattern_parts.append(f"[文档{i+1}]")
                    if i < len(eos_positions):
                        pattern_parts.append(f"{repr(eos_token_str)}")
                
                if len(doc_lengths) > 3:
                    pattern_parts.append("...")
                
                print(f"      {' '.join(pattern_parts)}")
    
    print("\n" + "=" * 100)
    print("✅ 数据检查完成")
    print("=" * 100 + "\n")


def inspect_first_batch(dataloader, tokenizer, num_samples: int = 2) -> None:
    """从 DataLoader 获取第一个 batch 并检查。
    
    Args:
        dataloader: DataLoader 对象
        tokenizer: Tokenizer 对象
        num_samples: 要显示的样本数量
    """
    print("\n🔍 正在获取第一个 batch 进行检查...")
    
    try:
        batch = next(iter(dataloader))
        inspect_batch(batch, tokenizer, max_samples=num_samples)
    except StopIteration:
        print("❌ DataLoader 为空，无法获取 batch")
    except Exception as e:
        print(f"❌ 检查 batch 时出错: {e}")
        import traceback
        traceback.print_exc()
