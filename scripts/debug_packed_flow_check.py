#!/usr/bin/env python3
"""Packed-mode end-to-end sanity check with explicit shape/semantics validation.

What this script validates (with clear EXPECT vs RESULT per step):
- A) Synthetic packed examples: input_ids/doc_ids per position (B=2, T=8)
- B) Collate (packed):
  - attention_mask shape is (B,T,T) and semantics = causal AND same-doc
  - labels = input_ids except EOS positions set to -100
  - position_ids exists and is (B,T)
- C) Convert 3D attention_mask -> 2D for embeddings via diagonal (B,T)
  - EXPECT: diagonal is True for all non-padding tokens
- D) Model forward with 3D mask kept for attention and 2D mask used by embeddings
  - EXPECT: forward succeeds; logits shape (B,T,V); loss is finite

Run:
  .venv/bin/python scripts/debug_packed_flow_check.py
"""

from __future__ import annotations

import math
import sys
from typing import List, Tuple

import torch

from src.dataset.base import DataConfig
from src.dataset.pretrain import PretrainConfigExtras, PretrainDatasetModule
from src.models.gpt2.model import GPT2Model


class DummyTokenizer:
    def __init__(self, eos_token_id: int = 99, pad_token_id: int = 0) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token = "<pad>"


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _expect_result(expect: str, result: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[CHECK] {expect}\n       -> RESULT: {result}\n       -> {status}")


def build_synthetic_examples(B: int = 2, T: int = 8, eos_id: int = 99):
    """Construct minimal packed-like examples with explicit doc switches and EOS."""
    # Example 0: two docs split at pos=3 and pos=7 (EOS at 3 and 7)
    ex0_ids = torch.tensor([10, 11, 12, eos_id, 20, 21, 22, eos_id], dtype=torch.long)
    ex0_doc = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    # Example 1: split at pos=2 and pos=6 (EOS at 2 and 6)
    ex1_ids = torch.tensor([30, 31, eos_id, 40, 41, 42, eos_id, 50], dtype=torch.long)
    ex1_doc = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    examples = [
        {"input_ids": ex0_ids, "doc_ids": ex0_doc, "source": "synthetic"},
        {"input_ids": ex1_ids, "doc_ids": ex1_doc, "source": "synthetic"},
    ]
    return examples


def validate_attention_mask_semantics(
    attn: torch.Tensor, doc_ids: torch.Tensor
) -> Tuple[bool, int]:
    """Check (B,T,T) mask equals (causal AND same_doc). Returns (ok, mismatches)."""
    B, T, S = attn.shape
    mismatches = 0
    for b in range(B):
        for tgt in range(T):
            for src in range(S):
                expected = (src <= tgt) and (doc_ids[b, tgt] == doc_ids[b, src])
                got = bool(attn[b, tgt, src].item())
                if expected != got:
                    mismatches += 1
    return (mismatches == 0), mismatches


def main() -> None:
    _print_header("A) Synthetic examples (EXPECT: shapes (T,), doc_ids non-decreasing)")
    tok = DummyTokenizer(eos_token_id=99, pad_token_id=0)
    examples = build_synthetic_examples(eos_id=tok.eos_token_id)
    for i, ex in enumerate(examples):
        ids, docs = ex["input_ids"], ex["doc_ids"]
        print(f"Example {i}: input_ids shape={tuple(ids.shape)} values={ids.tolist()}")
        print(f"           doc_ids   shape={tuple(docs.shape)} values={docs.tolist()}")
        nondec = torch.all(docs[1:] - docs[:-1] >= 0).item()
        _expect_result(
            "doc_ids must be non-decreasing",
            f"non-decreasing={nondec}",
            ok=bool(nondec),
        )

    _print_header("B) Collate (packed) -> attention_mask (B,T,T), labels, position_ids")
    module = PretrainDatasetModule(
        config=DataConfig(sources=[], tokenizer_batch_size=2048, num_workers=0, prefetch_factor=2, pin_memory=True),
        tokenizer=tok,
        extras=PretrainConfigExtras(
            sequence_length=8,
            add_bos=False,
            add_eos=False,
            pad_to_multiple_of=None,
            padding_strategy="do_not_pad",
            pack_sequences=True,
            shuffle_buffer_size=128,
        ),
        seed=42,
    )

    batch = module.collate_fn(examples)
    input_ids = batch["input_ids"]
    doc_ids = torch.stack([ex["doc_ids"] for ex in examples])
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]
    labels = batch["labels"]

    print(f"input_ids      shape={tuple(input_ids.shape)} (EXPECT: (2,8))")
    print(f"attention_mask shape={tuple(attention_mask.shape)} (EXPECT: (2,8,8))")
    print(f"position_ids   shape={tuple(position_ids.shape)} (EXPECT: (2,8))")
    print(f"labels         shape={tuple(labels.shape)} (EXPECT: (2,8))")

    ok_mask, mism = validate_attention_mask_semantics(attention_mask, doc_ids)
    _expect_result(
        "attention_mask == causal AND same_doc for all (tgt,src)",
        f"mismatches={mism}",
        ok=ok_mask,
    )

    # Labels check: EOS positions must be -100; others equal to input_ids
    eos_id = tok.eos_token_id
    eos_pos = (input_ids == eos_id)
    eos_ok = torch.all(labels[eos_pos] == -100).item() if eos_pos.any() else True
    same_elsewhere = torch.all(labels[~eos_pos] == input_ids[~eos_pos]).item()
    _expect_result(
        "labels: EOS positions set to -100",
        f"eos_ok={eos_ok}",
        ok=bool(eos_ok),
    )
    _expect_result(
        "labels: non-EOS positions equal to input_ids",
        f"equal={same_elsewhere}",
        ok=bool(same_elsewhere),
    )

    _print_header("C) Embeddings mask (2D) derived from diagonal of 3D mask")
    diag = torch.diagonal(attention_mask, dim1=-2, dim2=-1)
    print(f"diag(attention_mask) shape={tuple(diag.shape)} (EXPECT: (2,8))")
    all_true = torch.all(diag).item()
    _expect_result(
        "diagonal should be True for all non-padding tokens (no padding in synthetic)",
        f"all_true={all_true}",
        ok=bool(all_true),
    )

    _print_header("D) Model forward with 3D attention_mask (embeddings use 2D diag internally)")
    # Tiny GPT-2 model to keep the test lightweight
    model = GPT2Model(
        vocab_size=200,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=16,
        attn_dropout=0.0,
        resid_dropout=0.0,
        qkv_bias=True,
        use_flash=False,
        pad_token_id=0,
        output_hidden_states=False,
        output_attentions=False,
    )

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, labels=labels)
    logits_shape = tuple(out.logits.shape)
    loss_ok = (out.loss is not None) and (not math.isnan(out.loss.item()))
    print(f"logits shape={logits_shape} (EXPECT: (2,8,200))")
    print(f"loss is finite: {loss_ok}")
    _expect_result("forward completed without shape errors", "ok=True", ok=True)

    _print_header("Summary")
    print("All staged checks finished. If any FAIL above, inspect corresponding step.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
            print("ERROR:", e)
            raise
