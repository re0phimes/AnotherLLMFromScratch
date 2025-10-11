from __future__ import annotations

"""Direct Preference Optimization (DPO) dataset module.

Pipeline context:
    - 所在步骤：Trainer 之前的数据处理末端，将偏好比较样本 (prompt, chosen, rejected)
      转换为模型可直接消费的张量。
    - 输入形式：
        * 本地 JSONL：每行含 `prompt`、`chosen`、`rejected` 字段（或配置项对应字段名）。
        * streaming 数据：Hugging Face datasets record，需返回等价的键值对。
    - 输出形式：分别编码 prompt/chosen/rejected 的 `input_ids` 和 `attention_mask`，并
      提供长度 metadata，Trainer 可直接传给模型计算 DPO 损失。
    - 目标效果：集中管理 tokenizer/padding 策略，确保三类序列在 batch 中对齐，方便
      上层进行对数概率比较。

Workflow summary:
    1. 训练配置在 `configs/train/*.yaml` 中指定偏好数据集信息，调用
       `DPODatasetModule.from_config`。
    2. 解析后的 DataConfig 可包含多个数据源（local JSONL / streaming），每条样本
       需包含 prompt / chosen / rejected 字段。
    3. 模块使用传入的 tokenizer 对三段文本编码，返回对齐的批次，供 DPO 损失函数
       计算 preference log-prob 差值。

Expected batch schema produced by `collate_fn`:
    {
        "prompt_input_ids": Tensor[batch, seq_len],
        "prompt_attention_mask": Tensor[batch, seq_len],
        "chosen_input_ids": Tensor[batch, seq_len],
        "chosen_attention_mask": Tensor[batch, seq_len],
        "rejected_input_ids": Tensor[batch, seq_len],
        "rejected_attention_mask": Tensor[batch, seq_len],
        "metadata": {
            "source": List[str],
            "prompt_length": List[int],
            "chosen_length": List[int],
            "rejected_length": List[int],
        }
    }

Example (`batch_size=1`, GPT-2 tokenizer, `seq_len=10`):
    - prompt: "Q: Who wrote Sherlock Holmes?" → tokens `[1212, 25, 1639, ...]`
    - chosen: "A: Arthur Conan Doyle." → tokens `[64, 25, 1460, ...]`
    - rejected: "A: Sherlock wrote Sherlock." → tokens `[64, 25, 1639, ...]`
    - 对应的 `*_attention_mask` 为 1 表示有效 token，padding 为 0。
    - `metadata`:
        {
            "source": ["sample_001"],
            "prompt_length": [7],
            "chosen_length": [6],
            "rejected_length": [6]
        }

本模块不负责在线偏好数据生成，仅消费离线样本；清洗/构造逻辑应放在 scripts/ 中。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset

from .base import BaseDatasetModule, DataConfig, DataSourceConfig, parse_data_config

try:
    import datasets as hf_datasets
except ImportError:  # pragma: no cover
    hf_datasets = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper structures
# ---------------------------------------------------------------------------

@dataclass
class DPOConfigExtras:
    sequence_length: int
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    separator: str = "\n\n"
    padding_strategy: str = "max_length"


def _extract_preference_triplet(
    record: Mapping[str, Any],
    *,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
    separator: str,
) -> Optional[Tuple[str, str, str]]:
    prompt = record.get(prompt_key, "")
    chosen = record.get(chosen_key, "")
    rejected = record.get(rejected_key, "")

    if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    prompt = prompt.strip()
    chosen = chosen.strip()
    rejected = rejected.strip()
    if not prompt or not chosen or not rejected:
        return None
    prompt = prompt.replace("\r\n", "\n")
    chosen = chosen.replace("\r\n", "\n")
    rejected = rejected.replace("\r\n", "\n")
    prompt_with_sep = f"{prompt}{separator}" if separator else prompt
    return prompt_with_sep, chosen, rejected


class LocalDPOJsonlDataset(Dataset[Dict[str, Any]]):
    """本地 JSONL 偏好数据集。"""

    def __init__(self, source: DataSourceConfig, extras: DPOConfigExtras) -> None:
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"Local data file not found: {source.path}")
        self._records: List[Dict[str, str]] = []
        max_samples = source.max_samples
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                triplet = _extract_preference_triplet(
                    obj,
                    prompt_key=extras.prompt_key,
                    chosen_key=extras.chosen_key,
                    rejected_key=extras.rejected_key,
                    separator=extras.separator,
                )
                if not triplet:
                    continue
                prompt, chosen, rejected = triplet
                self._records.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": source.name,
                })
                if max_samples is not None and len(self._records) >= max_samples:
                    break
        if not self._records:
            raise ValueError(f"Local DPO dataset {source.path} yielded no usable samples.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self._records[index]


class StreamingDPODataset(IterableDataset):
    """Hugging Face datasets streaming 偏好数据读取。"""

    def __init__(self, source: DataSourceConfig, extras: DPOConfigExtras) -> None:
        if hf_datasets is None:  # pragma: no cover
            raise ImportError("datasets package is required for streaming mode.")
        self.source = source
        self.extras = extras

    def __iter__(self) -> Iterator[Dict[str, str]]:
        split = self.source.streaming_kwargs.get("split", "train")
        stream = hf_datasets.load_dataset(
            path=self.source.path,
            name=self.source.subset,
            split=split,
            streaming=True,
            **{k: v for k, v in self.source.streaming_kwargs.items() if k != "split"},
        )
        iterator: Iterable[Any] = stream.shuffle(seed=42) if self.source.shuffle else stream
        counter = 0
        for sample in iterator:
            if not isinstance(sample, Mapping):
                continue
            triplet = _extract_preference_triplet(
                sample,
                prompt_key=self.extras.prompt_key,
                chosen_key=self.extras.chosen_key,
                rejected_key=self.extras.rejected_key,
                separator=self.extras.separator,
            )
            if not triplet:
                continue
            prompt, chosen, rejected = triplet
            yield {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": self.source.name,
            }
            counter += 1
            if self.source.max_samples is not None and counter >= self.source.max_samples:
                break


def _replicate_with_weights(datasets: Sequence[Dataset[Any]], sources: Sequence[DataSourceConfig]) -> Dataset[Any]:
    min_weight = min(src.sampling_weight for src in sources)
    expanded: List[Dataset[Any]] = []
    for dataset, source in zip(datasets, sources):
        multiplier = max(1, int(round(source.sampling_weight / min_weight)))
        expanded.extend([dataset] * multiplier)
    return ConcatDataset(expanded) if len(expanded) > 1 else expanded[0]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class DPODatasetModule(BaseDatasetModule):
    """偏好比较数据模块，生成 prompt-chosen-rejected 批次。"""

    def __init__(
        self,
        *,
        config: DataConfig,
        tokenizer,
        extras: DPOConfigExtras,
        seed: int = 42,
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, seed=seed)
        self.extras = extras
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        self.sequence_length = extras.sequence_length

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        tokenizer,
        seed: int = 42,
    ) -> "DPODatasetModule":
        data_config = parse_data_config(cfg)
        sequence_length = int(cfg.get("sequence_length") or getattr(tokenizer, "model_max_length", 1024) or 1024)
        extras = DPOConfigExtras(
            sequence_length=sequence_length,
            prompt_key=str(cfg.get("prompt_key", "prompt")),
            chosen_key=str(cfg.get("chosen_key", "chosen_response" if "chosen_response" in cfg else "chosen")),
            rejected_key=str(cfg.get("rejected_key", "rejected_response" if "rejected_response" in cfg else "rejected")),
            separator=str(cfg.get("separator", "\n\n")),
            padding_strategy=str(cfg.get("padding", "max_length")),
        )
        return cls(config=data_config, tokenizer=tokenizer, extras=extras, seed=seed)

    def build_dataset(self) -> Dataset[Any]:
        datasets: List[Dataset[Any]] = []
        for source in self.config.sources:
            if source.type == "local":
                datasets.append(LocalDPOJsonlDataset(source, self.extras))
            elif source.type == "streaming":
                datasets.append(StreamingDPODataset(source, self.extras))
            else:  # pragma: no cover
                raise ValueError(f"Unknown data source type {source.type}")
        if not datasets:
            raise ValueError("No datasets were constructed.")
        if len(datasets) == 1:
            return datasets[0]
        return _replicate_with_weights(datasets, self.config.sources)

    def collate_fn(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        prompts = [str(ex["prompt"]) for ex in examples]
        chosen = [str(ex["chosen"]) for ex in examples]
        rejected = [str(ex["rejected"]) for ex in examples]
        source_names = [str(ex.get("source", "unknown")) for ex in examples]

        padding = self._resolve_padding_strategy()
        prompt_encoded = self.tokenizer(
            prompts,
            add_special_tokens=True,
            max_length=self.sequence_length,
            truncation=True,
            padding=padding,
            return_attention_mask=True,
            return_tensors="pt",
        )
        chosen_encoded = self.tokenizer(
            chosen,
            add_special_tokens=True,
            max_length=self.sequence_length,
            truncation=True,
            padding=padding,
            return_attention_mask=True,
            return_tensors="pt",
        )
        rejected_encoded = self.tokenizer(
            rejected,
            add_special_tokens=True,
            max_length=self.sequence_length,
            truncation=True,
            padding=padding,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "prompt_input_ids": prompt_encoded["input_ids"],
            "prompt_attention_mask": prompt_encoded["attention_mask"],
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_attention_mask": rejected_encoded["attention_mask"],
            "metadata": {
                "source": source_names,
                "prompt_length": [len(ids) for ids in prompt_encoded["input_ids"]],
                "chosen_length": [len(ids) for ids in chosen_encoded["input_ids"]],
                "rejected_length": [len(ids) for ids in rejected_encoded["input_ids"]],
            },
        }

    def _resolve_padding_strategy(self) -> str:
        strategy = self.extras.padding_strategy
        if strategy == "max_length":
            return "max_length"
        if strategy == "longest":
            return "longest"
        if strategy == "do_not_pad":
            return "do_not_pad"
        raise ValueError("padding must be one of {'max_length','longest','do_not_pad'}")


__all__ = ["DPODatasetModule"]

# Notes:
#   - 用于 DPO / PairRM / RRHF 等偏好比较任务，依赖外部 Trainer 计算损失。
#   - 若数据包含额外上下文字段（如系统 prompt），可在 _extract_preference_triplet
#     中拼接到 prompt，或扩展返回的 metadata。
#   - 需要注意 chosen/rejected 序列长度差异过大时，可能需要在 tokenizer
#     层做截断策略优化，以减少无意义的 padding。
