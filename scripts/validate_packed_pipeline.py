#!/usr/bin/env python3
"""Validate packed pretraining data pipeline end-to-end.

Stages checked:
1. Raw packed dataset output (`input_ids`, `doc_ids`).
2. Collate function output (`input_ids`, `attention_mask`, `labels`).
3. Consistency assertions between raw tokens and collated batches.

Usage:
    python scripts/validate_packed_pipeline.py \
        --config configs/train/gpt2_pretrain_packed.yaml \
        --batches 3 --batch-size 2 --max-samples 2000

The script raises AssertionError on any inconsistency and prints
per-stage diagnostics for manual inspection.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.pretrain import (  # noqa: E402
    PretrainConfigExtras,
    PretrainDatasetModule,
    parse_data_config,
)


def _load_data_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as fp:
        full_cfg = yaml.safe_load(fp)
    if "data" not in full_cfg:
        raise KeyError("config must contain a 'data' section")
    return full_cfg["data"]


def _override_max_samples(data_cfg: Dict, max_samples: int | None) -> None:
    if max_samples is None:
        return
    if "data_sources" in data_cfg:
        for entry in data_cfg["data_sources"]:
            entry["max_samples"] = max_samples
    else:
        data_cfg["max_samples"] = max_samples


def _collect_raw_batch(dataset_iter, batch_size: int) -> List[Dict[str, torch.Tensor]]:
    batch: List[Dict[str, torch.Tensor]] = []
    for _ in range(batch_size):
        example = next(dataset_iter)
        if "input_ids" not in example or "doc_ids" not in example:
            raise KeyError("Packed dataset example must contain 'input_ids' and 'doc_ids'")
        batch.append(example)
    return batch


def _token_str(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if token == tokenizer.eos_token:
        return f"<EOS:{token_id}>"
    return token


def _pretty_print_example(
    *,
    tokenizer,
    example: Dict[str, torch.Tensor],
    example_idx: int,
    doc_boundaries: Sequence[int],
    max_positions: int,
) -> None:
    input_ids = example["input_ids"].tolist()
    doc_ids = example["doc_ids"].tolist()

    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Raw example {example_idx} decoded text:\n{decoded_full}\n")

    header = "pos | token_id | token | doc_id"
    print(header)
    print("-" * len(header))

    limit = min(len(input_ids), max_positions)
    for pos in range(limit):
        token_id = input_ids[pos]
        doc_id = doc_ids[pos]
        token_repr = _token_str(tokenizer, token_id)
        marker = "*" if pos in doc_boundaries else " "
        print(f"{pos:3d}{marker}| {token_id:8d} | {token_repr} | {doc_id}")
    if limit < len(input_ids):
        print(f"... (truncated to first {limit} positions)")


def _validate_raw_examples(
    *,
    tokenizer,
    examples: Sequence[Dict[str, torch.Tensor]],
    seq_len: int,
    max_positions: int,
) -> None:
    for idx, example in enumerate(examples):
        input_ids = example["input_ids"]
        doc_ids = example["doc_ids"]
        if input_ids.shape != (seq_len,) or doc_ids.shape != (seq_len,):
            raise AssertionError(
                f"Example {idx}: expected shape ({seq_len},) got {input_ids.shape} and {doc_ids.shape}"
            )
        if not torch.all(doc_ids >= 0):
            raise AssertionError(f"Example {idx}: doc_ids must be non-negative")
        # doc_ids should be non-decreasing and only change at document boundaries
        diffs = doc_ids[1:] - doc_ids[:-1]
        if not torch.all((diffs == 0) | (diffs > 0)):
            raise AssertionError(f"Example {idx}: doc_ids must be non-decreasing")
        # check carry-over is contiguous when doc_ids stable
        boundaries = (diffs != 0).nonzero(as_tuple=False).flatten().tolist()
        print(f"Raw example {idx}: doc boundaries at positions {boundaries}")
        _pretty_print_example(
            tokenizer=tokenizer,
            example=example,
            example_idx=idx,
            doc_boundaries=boundaries,
            max_positions=max_positions,
        )


def _validate_collated_batch(
    collated: Dict[str, torch.Tensor],
    raw_examples: Sequence[Dict[str, torch.Tensor]],
    tokenizer_eos_id: int | None,
) -> None:
    input_ids = collated["input_ids"]
    attention_mask = collated["attention_mask"]
    labels = collated["labels"]

    batch_size, seq_len = input_ids.shape

    # Assert shapes
    if attention_mask.shape != (batch_size, seq_len, seq_len):
        raise AssertionError(
            f"attention_mask must have shape (B, T, T), got {attention_mask.shape}"
        )
    if labels.shape != (batch_size, seq_len):
        raise AssertionError(f"labels must have shape (B, T), got {labels.shape}")
    if attention_mask.dtype is not torch.bool:
        raise AssertionError(f"attention_mask dtype must be bool, got {attention_mask.dtype}")

    for batch_idx, raw in enumerate(raw_examples):
        raw_ids = raw["input_ids"]
        raw_doc = raw["doc_ids"]
        if not torch.equal(input_ids[batch_idx], raw_ids):
            raise AssertionError(f"Batch {batch_idx}: input_ids mismatch with raw example")
        # Labels must align with input ids except EOS positions
        diff_mask = labels[batch_idx] != input_ids[batch_idx]
        if tokenizer_eos_id is None:
            if diff_mask.any():
                raise AssertionError(f"Batch {batch_idx}: label differs without eos_token")
        else:
            eos_positions = input_ids[batch_idx] == tokenizer_eos_id
            if not torch.equal(diff_mask, eos_positions):
                raise AssertionError(
                    f"Batch {batch_idx}: labels must differ only at EOS positions"
                )
            if not torch.all(labels[batch_idx][eos_positions] == -100):
                raise AssertionError(
                    f"Batch {batch_idx}: EOS labels must be -100"
                )
        # Validate attention mask respects causal + document constraints
        mask = attention_mask[batch_idx]
        for tgt, src in itertools.product(range(seq_len), repeat=2):
            allowed = mask[tgt, src].item()
            same_doc = raw_doc[tgt] == raw_doc[src]
            if src > tgt and allowed:
                raise AssertionError(
                    f"Batch {batch_idx}: mask[{tgt},{src}] should be 0 for future tokens"
                )
            if same_doc:
                if src <= tgt and not allowed:
                    raise AssertionError(
                        f"Batch {batch_idx}: same doc positions must attend ({tgt},{src})"
                    )
            else:
                if allowed:
                    raise AssertionError(
                        f"Batch {batch_idx}: cross-doc positions must be masked ({tgt},{src})"
                    )


def run_validation(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    data_cfg = _load_data_config(config_path)
    _override_max_samples(data_cfg, args.max_samples)

    tokenizer_name = args.tokenizer or "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Ensure pad token for packed mode (dataset module expects it)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    extras = PretrainConfigExtras(
        sequence_length=args.sequence_length,
        add_bos=data_cfg.get("add_bos", False),
        add_eos=data_cfg.get("add_eos", False),
        pad_to_multiple_of=data_cfg.get("pad_to_multiple_of"),
        padding_strategy=str(data_cfg.get("padding", "do_not_pad")),
        pack_sequences=True,
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 5000)),
    )

    dataset_module = PretrainDatasetModule(
        config=parse_data_config(data_cfg),
        tokenizer=tokenizer,
        extras=extras,
        seed=args.seed,
    )

    dataset = dataset_module.build_dataset()
    dataset_iter = iter(dataset)

    print("== Packed Dataset Validation ==")
    for batch_idx in range(args.batches):
        print(f"\n-- Batch {batch_idx} --")
        raw_examples = _collect_raw_batch(dataset_iter, args.batch_size)
        _validate_raw_examples(
            tokenizer=tokenizer,
            examples=raw_examples,
            seq_len=args.sequence_length,
            max_positions=args.print_max_positions,
        )
        collated = dataset_module.collate_fn(raw_examples)
        _validate_collated_batch(collated, raw_examples, tokenizer.eos_token_id)
        print(
            f"Batch {batch_idx}: input_ids shape {collated['input_ids'].shape}, "
            f"labels -100 count {int((collated['labels'] == -100).sum().item())}"
        )

    print("\nAll requested batches validated successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate packed data pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs/train/gpt2_pretrain_packed.yaml"),
        help="Training config with pack_sequences enabled.",
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name or path")
    parser.add_argument("--sequence-length", type=int, default=1024, help="Sequence length to validate")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of packed sequences per batch")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches to inspect")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Limit samples read from each data source for faster validation (None for unlimited)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for packed dataset")
    parser.add_argument(
        "--print-max-positions",
        type=int,
        default=64,
        help="Maximum token positions to print per example",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
