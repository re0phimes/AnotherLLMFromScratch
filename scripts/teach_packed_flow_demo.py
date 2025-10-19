#!/usr/bin/env python3
"""
教学版：打包式（packed）预训练数据从原始文本到模型前向的完整演示与校验

包含阶段：
1) 原始文本 -> 分词(tokenize) -> 每段文档末尾追加 EOS
2) 文档打包：拼接两个文档为一条 token 流，并生成 doc_ids 序列
3) collate（packed 专用）：构造 (B,T,T) 的注意力掩码 attention_mask，以及 labels/position_ids
4) 嵌入的 2D 掩码：从 3D 掩码的对角线提取 (B,T)
5) 模型前向：保持注意力使用 3D 掩码，嵌入使用 2D 掩码（脚本内模型实现已兼容）

每一步都会打印：
- 正在校验的内容
- 期望的结果（EXPECT）
- 实际结果（RESULT）

运行：
  python scripts/teach_packed_flow_demo.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from src.dataset.base import DataConfig
from src.dataset.pretrain import PretrainConfigExtras, PretrainDatasetModule
from src.models.gpt2.model import GPT2Model


# ========== 简单 Tokenizer（空格切分，稳定词表） ==========
class SimpleTokenizer:
    def __init__(self, eos_token_id: int = 999, pad_token_id: int = 0) -> None:
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.next_id: int = 1  # 预留 0 给 PAD
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def _id(self, token: str) -> int:
        if token not in self.word2id:
            tid = self.next_id
            self.word2id[token] = tid
            self.id2word[tid] = token
            self.next_id += 1
        return self.word2id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        # 教学目的：以空格切分，逐词映射到 id，不自动加特殊符号
        toks = [t for t in text.strip().split() if t]
        return [self._id(t) for t in toks]


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def expect_result(expect: str, result: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"- CHECK: {expect}\n  RESULT: {result}\n  ==> {status}")


def build_example_from_docs(tokenizer: SimpleTokenizer, docs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """将多段文档打包为一条 token 流，并返回 (input_ids, doc_ids)。

    规则：
    - 文档逐段 encode 后在末尾追加 EOS（eos_token_id）
    - doc_ids 在同一文档（含该文档的 EOS）保持常数，跨文档递增
    """
    ids: List[int] = []
    docs_arr: List[int] = []
    doc_idx = 0
    for text in docs:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(token_ids)
        ids.append(tokenizer.eos_token_id)
        docs_arr.extend([doc_idx] * (len(token_ids) + 1))
        doc_idx += 1
    return torch.tensor(ids, dtype=torch.long), torch.tensor(docs_arr, dtype=torch.long)


def validate_attention_mask(attn: torch.Tensor, doc_ids: torch.Tensor) -> Tuple[bool, int]:
    """验证 (B,T,T) 掩码是否等于 (因果下三角 AND 同文档)。"""
    B, T, S = attn.shape
    mismatches = 0
    for b in range(B):
        for tgt in range(T):
            for src in range(S):
                expected = (src <= tgt) and (doc_ids[b, tgt] == doc_ids[b, src])
                got = bool(attn[b, tgt, src].item())
                if expected != got:
                    mismatches += 1
    return mismatches == 0, mismatches


def main() -> None:
    # 1) 原始文本（两条样本，每条样本内部包含 2 段文档）
    print_header("1) 原始文本与分词（EXPECT: 打印每段文档与其 token 映射，EOS 追加在段尾）")
    tok = SimpleTokenizer(eos_token_id=999, pad_token_id=0)
    raw_samples = [
        ["Hello A", "World B"],
        ["Foo Bar", "Baz Qux"],
    ]

    examples: List[Dict[str, torch.Tensor]] = []
    for i, docs in enumerate(raw_samples):
        print(f"样本 {i}：")
        for j, d in enumerate(docs):
            ids = tok.encode(d)
            print(f"  文档 {j} 原文: {d}")
            print(f"  文档 {j} 分词ID: {ids} （不含 EOS）")
        input_ids_1d, doc_ids_1d = build_example_from_docs(tok, docs)
        print(f"  拼接后 input_ids(+EOS): {input_ids_1d.tolist()}")
        print(f"  对应 doc_ids:          {doc_ids_1d.tolist()} （同一文档含 EOS）")
        examples.append({"input_ids": input_ids_1d, "doc_ids": doc_ids_1d, "source": "teaching"})

    # 2) collate（packed）：产生 (B,T,T) attention_mask, (B,T) position_ids, (B,T) labels
    print_header("2) collate（packed）生成 batch（EXPECT: attention_mask 为 (B,T,T)，position/labels 为 (B,T)）")
    T = max(ex["input_ids"].numel() for ex in examples)
    # 若需要，可 pad/trunc 到统一长度 T；这里两条样本长度相同，无需 pad
    B = len(examples)

    module = PretrainDatasetModule(
        config=DataConfig(sources=[], tokenizer_batch_size=2048, num_workers=0, prefetch_factor=2, pin_memory=True),
        tokenizer=tok,
        extras=PretrainConfigExtras(
            sequence_length=T,
            add_bos=False,
            add_eos=False,
            pad_to_multiple_of=None,
            padding_strategy="do_not_pad",
            pack_sequences=True,
            shuffle_buffer_size=16,
        ),
        seed=42,
    )

    batch = module.collate_fn(examples)
    input_ids = batch["input_ids"]  # (B,T)
    attention_mask = batch["attention_mask"]  # (B,T,T)
    position_ids = batch["position_ids"]  # (B,T)
    labels = batch["labels"]  # (B,T)
    doc_ids = torch.stack([ex["doc_ids"] for ex in examples])

    print(f"B={B}, T={T}")
    print(f"input_ids shape={tuple(input_ids.shape)}  EXPECT=(B,T)")
    print(f"attention_mask shape={tuple(attention_mask.shape)}  EXPECT=(B,T,T)")
    print(f"position_ids shape={tuple(position_ids.shape)}  EXPECT=(B,T)")
    print(f"labels shape={tuple(labels.shape)}  EXPECT=(B,T)")

    ok_mask, mism = validate_attention_mask(attention_mask, doc_ids)
    expect_result(
        "attention_mask 逐元素等于(因果下三角 AND 同文档)",
        f"mismatches={mism}",
        ok=ok_mask,
    )

    eos_mask = (input_ids == tok.eos_token_id)
    eos_ok = (labels[eos_mask] == -100).all().item() if eos_mask.any() else True
    same_else = (labels[~eos_mask] == input_ids[~eos_mask]).all().item()
    expect_result("labels 在 EOS 位置为 -100", f"eos_ok={eos_ok}", ok=bool(eos_ok))
    expect_result("labels 非 EOS 位置等于 input_ids", f"equal={same_else}", ok=bool(same_else))

    # 展示一个样本的 (T,T) 掩码矩阵（小尺寸便于可视化）
    for b in range(B):
        print(f"\n样本 {b} 的 attention_mask(前 8x8 或完整)：")
        m = attention_mask[b]
        view = m[: min(8, T), : min(8, T)].int().tolist()
        for row in view:
            print("  ", row)

    # 3) 嵌入层需要 (B,T) 掩码 -> 从 3D 掩码取对角线
    print_header("3) 嵌入使用的 2D 掩码（来自 3D 掩码对角线）")
    diag2d = torch.diagonal(attention_mask, dim1=-2, dim2=-1)  # (B,T)
    print(f"diag(attention_mask) shape={tuple(diag2d.shape)}  EXPECT=(B,T)")
    # 这里无 padding，期望对角线全 True
    expect_result("无 padding 时对角线应全为 True", f"all_true={bool(diag2d.all().item())}", ok=bool(diag2d.all().item()))

    # 4) 模型前向：注意力仍用 3D 掩码；嵌入内部会接受 2D 掩码（仓库代码已做兼容）
    print_header("4) 模型前向（EXPECT: logits 形状为 (B,T,V)，loss 为有限值）")
    V = max(int(input_ids.max().item()) + 1, tok.eos_token_id + 1, 200)  # 保证词表覆盖
    model = GPT2Model(
        vocab_size=V,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=max(T, 16),
        attn_dropout=0.0,
        resid_dropout=0.0,
        qkv_bias=True,
        use_flash=False,
        pad_token_id=tok.pad_token_id,
        output_hidden_states=False,
        output_attentions=False,
    )

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits_shape = tuple(out.logits.shape)
    loss_val = out.loss.item() if out.loss is not None else float("nan")
    expect_result("前向完成且 logits 维度正确", f"logits_shape={logits_shape}", ok=(logits_shape == (B, T, V)))
    expect_result("loss 有限", f"loss={loss_val}", ok=(not math.isnan(loss_val)))

    print_header("完成：若以上均 PASS，则 packed 流程各阶段语义与形状均符合预期。")


if __name__ == "__main__":
    main()
