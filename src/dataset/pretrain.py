from __future__ import annotations

"""Pretraining dataset module.

Pipeline context:
    - 所在步骤：数据管线的最终处理环节，将清洗后的文本样本转换为模型可直接使用
      的张量批次。
    - 输入形式：
        * 本地 JSONL：每行 `{"text": "..."}` 或兼容字段（见 `_extract_text`）。
        * streaming 数据：Hugging Face datasets record，需至少包含 `text`/`content` 等
          文本字段。
    - 输出形式：`input_ids`、`attention_mask`、`labels`、`metadata`，供 Trainer 在
      训练循环中直接传入 `model(**batch)`。
    - 目标效果：集中处理 tokenizer/padding 逻辑，保持批次一致性，Trainer 与模型
      无需重新分词或手动构建掩码。

Workflow summary:
    1. Training入口读取 configs/train/*.yaml 中的 data 块，并传递给
       `PretrainDatasetModule.from_config`。
    2. 解析结果映射为多个 `DataSourceConfig`（local / streaming），初始化对应
       数据集迭代器。
    3. DataLoader 逐批获取原始文本 -> 使用传入的 tokenizer 编码 -> 输出预训练
       所需的 `input_ids/attention_mask/labels` 等张量。

Expected batch schema produced by `collate_fn`:
    {
        "input_ids": Tensor[batch, seq_len],
        "attention_mask": Tensor[batch, seq_len],
        "labels": Tensor[batch, seq_len],
        "metadata": {"source": List[str]}  # 每条样本来源名称
    }

Example (`batch_size=2`, `seq_len=6`, using GPT-2 tokenizer IDs):
    - 原始文本: ["Hello world", "Foo bar"]
    - `input_ids`:
        tensor([[15496,  995,   13, 50256, 50256, 50256],
                [  810,  348,   13, 50256, 50256, 50256]])
    - `attention_mask`:
        tensor([[1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0]])
    - `labels` 与 `input_ids` 相同，但 padding 位置被置为 -100：
        tensor([[15496,  995,   13, -100, -100, -100],
                [  810,  348,   13, -100, -100, -100]])

本模块仅负责离线语料 → token 序列的转换；任何原始清洗、下载脚本请放在
`scripts/` 或独立数据准备流程中。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset

from .base import BaseDatasetModule, DataConfig, DataSourceConfig, parse_data_config

try:
    import datasets as hf_datasets
except ImportError:  # pragma: no cover - streaming 功能依赖 datasets
    hf_datasets = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfigExtras:
    sequence_length: int
    add_bos: bool
    add_eos: bool
    pad_to_multiple_of: Optional[int]
    padding_strategy: str


def _extract_text(record: Any) -> Optional[str]:
    """尽可能从多种结构中抽取文本字段。"""
    if record is None:
        return None
    if isinstance(record, str):
        return record.strip()
    if isinstance(record, Mapping):
        for key in ("text", "content", "completion", "response"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class LocalJsonlDataset(Dataset[Dict[str, Any]]):
    """简单的本地 JSONL Map-style 数据集。"""

    def __init__(self, source: DataSourceConfig) -> None:
        path = Path(source.path)
        if not path.exists():
            raise FileNotFoundError(f"Local data file not found: {source.path}")
        self._records: List[Dict[str, Any]] = []
        max_samples = source.max_samples
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = {"text": line}
                text = _extract_text(obj)
                if not text:
                    continue
                self._records.append({"text": text, "source": source.name})
                if max_samples is not None and len(self._records) >= max_samples:
                    break
        if not self._records:
            raise ValueError(f"Local dataset {source.path} yielded no usable samples.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._records[index]


class StreamingIterableDataset(IterableDataset):
    """Hugging Face datasets streaming 包装。"""

    def __init__(self, source: DataSourceConfig) -> None:
        if hf_datasets is None:  # pragma: no cover
            raise ImportError("datasets package is required for streaming mode.")
        self.source = source

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        split = self.source.streaming_kwargs.get("split", "train")
        streaming = hf_datasets.load_dataset(
            path=self.source.path,
            name=self.source.subset,
            split=split,
            streaming=True,
            **{k: v for k, v in self.source.streaming_kwargs.items() if k != "split"},
        )
        iterator: Iterable[Any] = streaming.shuffle(seed=42) if self.source.shuffle else streaming
        counter = 0
        for sample in iterator:
            text = _extract_text(sample)
            if not text:
                continue
            yield {"text": text, "source": self.source.name}
            counter += 1
            if self.source.max_samples is not None and counter >= self.source.max_samples:
                break


def _expand_datasets_with_weights(datasets: Sequence[Dataset[Any]], sources: Sequence[DataSourceConfig]) -> Dataset[Any]:
    """根据 sampling_weight 粗略扩充数据集，实现简单的权重混合。"""
    min_weight = min(src.sampling_weight for src in sources)
    expanded: List[Dataset[Any]] = []
    for dataset, source in zip(datasets, sources):
        multiplier = max(1, int(round(source.sampling_weight / min_weight)))
        expanded.extend([dataset] * multiplier)
    return ConcatDataset(expanded) if len(expanded) > 1 else expanded[0]


# ---------------------------------------------------------------------------
# Main module implementation
# ---------------------------------------------------------------------------

class PretrainDatasetModule(BaseDatasetModule):
    """离线自回归预训练数据模块。"""

    def __init__(
        self,
        *,
        config: DataConfig,
        tokenizer,
        extras: PretrainConfigExtras,
        seed: int = 42,
    ) -> None:
        super().__init__(config=config, tokenizer=tokenizer, seed=seed)
        self.extras = extras
        if tokenizer.pad_token_id is None:
            # 预训练通常要求有 pad_token；若无则复用 eos。
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        self.sequence_length = extras.sequence_length

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        tokenizer,
        seed: int = 42,
    ) -> "PretrainDatasetModule":
        data_config = parse_data_config(cfg)
        sequence_length = int(cfg.get("sequence_length") or getattr(tokenizer, "model_max_length", 1024) or 1024)
        if sequence_length <= 0 or sequence_length > 16384:
            raise ValueError("sequence_length must be within (0, 16384].")
        extras = PretrainConfigExtras(
            sequence_length=sequence_length,
            add_bos=bool(cfg.get("add_bos", True)),
            add_eos=bool(cfg.get("add_eos", True)),
            pad_to_multiple_of=cfg.get("pad_to_multiple_of"),
            padding_strategy=str(cfg.get("padding", "max_length")),
        )
        return cls(config=data_config, tokenizer=tokenizer, extras=extras, seed=seed)

    def build_dataset(self) -> Dataset[Any]:
        datasets: List[Dataset[Any]] = []
        for source in self.config.sources:
            if source.type == "local":
                datasets.append(LocalJsonlDataset(source))
            elif source.type == "streaming":
                datasets.append(StreamingIterableDataset(source))
            else:  # pragma: no cover - parse_data_config 已校验
                raise ValueError(f"Unknown data source type {source.type}")
        if not datasets:
            raise ValueError("No datasets were constructed.")
        if len(datasets) == 1:
            return datasets[0]
        return _expand_datasets_with_weights(datasets, self.config.sources)

    def collate_fn(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        texts = [str(ex["text"]) for ex in examples]
        source_names = [str(ex.get("source", "unknown")) for ex in examples]

        padding = self._resolve_padding_strategy()
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.sequence_length,
            truncation=True,
            padding=padding,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if self.extras.add_bos and self.tokenizer.bos_token_id is not None:
            # 如果 tokenizer 未自动添加 BOS，可在此处补充
            pass  # 大多数 tokenizer 会在 add_special_tokens=True 时处理

        if self.extras.add_eos and self.tokenizer.eos_token_id is not None:
            # 同理，确保序列以 eos 结尾；如 tokenizer 已处理则无需额外操作
            pass

        labels = input_ids.clone()
        # 对 padding 区域置 -100，确保不会因为填充 token 而产生梯度；这常与
        # torch.nn.CrossEntropyLoss(ignore_index=-100) 搭配使用。
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": {"source": source_names},
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


__all__ = ["PretrainDatasetModule"]
