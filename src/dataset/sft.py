from __future__ import annotations

"""Supervised fine-tuning (SFT) dataset module.

Pipeline context:
    - 所在步骤：Trainer 之前的数据处理末端，将指令/响应格式样本转为模型可直接
      使用的张量。
    - 输入形式：
        * 本地 JSONL：常见键 `instruction`/`input`/`output`（或配置里指定的自定义
          字段）。
        * streaming 数据：Hugging Face datasets record，需提供相同语义的键值对。
    - 输出形式：`input_ids`、`attention_mask`、`labels`（prompt 部分为 -100）、
      `metadata`，训练循环可直接 `model(**batch)`。
    - 目标效果：集中实现 prompt/response 拼接、tokenizer 批处理与 label masking，
      避免在 Trainer 或模型内部重复编码。

Workflow summary:
    1. 训练入口读取 configs/train/*.yaml 中 data/sft 块 → 调用
       `SFTDatasetModule.from_config(...)`。
    2. 基类解析 data 配置，构建一个或多个数据源（本地 JSONL / streaming）。
    3. `SFTDatasetModule` 将每条样本的 prompt / input / response 拼接成完整上下文，
       使用传入 tokenizer 编码，并生成只在响应阶段计算 loss 的 labels。

Expected batch schema produced by `collate_fn`:
    {
        "input_ids": Tensor[batch, seq_len],
        "attention_mask": Tensor[batch, seq_len],
        "labels": Tensor[batch, seq_len],  # prompt 部分被置为 -100
        "metadata": {
            "source": List[str],
            "prompt_length": List[int]  # 每条样本 prompt token 数
        }
    }

Example (`batch_size=1`, GPT-2 tokenizer, `seq_len=8`):
    - prompt: "Instruction:" → tokens `[1212, 25, 3894, 25]`
    - response: "Sure!" → tokens `[8258, 0, 0, 0]` (示例中 padding 到固定长度)
    - `input_ids`: tensor([[1212,   25, 3894,   25, 8258,     0,     0,     0]])
    - `attention_mask`: tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    - `labels`: tensor([[-100, -100, -100, -100, 8258, -100, -100, -100]])  # prompt 段置为 -100

本模块只消费已清洗好的 JSONL/streaming 数据；原始数据加工需在 scripts/ 中完成。
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
except ImportError:  # pragma: no cover - streaming 功能依赖 datasets
    hf_datasets = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper structures & utilities
# ---------------------------------------------------------------------------

@dataclass
class SFTConfigExtras:
    sequence_length: int
    prompt_key: str = "instruction"
    input_key: Optional[str] = "input"
    response_key: str = "output"
    separator: str = "\n\n"
    prompt_prefix: str = ""
    response_prefix: str = ""
    add_bos: bool = True
    add_eos: bool = True
    padding_strategy: str = "max_length"


def _normalize_prompt_response(
    record: Mapping[str, Any],
    *,
    prompt_key: str,
    input_key: Optional[str],
    response_key: str,
    separator: str,
    prompt_prefix: str,
    response_prefix: str,
) -> Optional[Tuple[str, str]]:
    """从通用指令格式中提取 prompt/response 文本。"""
    prompt_part = record.get(prompt_key, "")
    input_part = record.get(input_key, "") if input_key else ""
    response_part = record.get(response_key, "")

    if isinstance(prompt_part, str):
        prompt = prompt_part.strip()
    else:
        prompt = ""
    if isinstance(input_part, str) and input_part.strip():
        prompt = f"{prompt}{separator}{input_part.strip()}" if prompt else input_part.strip()
    if isinstance(response_part, str):
        response = response_part.strip()
    else:
        response = ""

    prompt = f"{prompt_prefix}{prompt}" if prompt_prefix else prompt
    response = f"{response_prefix}{response}" if response_prefix else response

    if not prompt or not response:
        return None
    return prompt, response


class LocalSFTJsonlDataset(Dataset[Dict[str, Any]]):
    """本地 JSONL Map-style 数据集，记录 prompt/response 字段。"""

    def __init__(self, source: DataSourceConfig, extras: SFTConfigExtras) -> None:
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
                    obj = {extras.prompt_key: "", extras.response_key: line}
                normalized = _normalize_prompt_response(
                    obj,
                    prompt_key=extras.prompt_key,
                    input_key=extras.input_key,
                    response_key=extras.response_key,
                    separator=extras.separator,
                    prompt_prefix=extras.prompt_prefix,
                    response_prefix=extras.response_prefix,
                )
                if not normalized:
                    continue
                prompt, response = normalized
                self._records.append({
                    "prompt": prompt,
                    "response": response,
                    "source": source.name,
                })
                if max_samples is not None and len(self._records) >= max_samples:
                    break
        if not self._records:
            raise ValueError(f"Local SFT dataset {source.path} yielded no usable samples.")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self._records[index]


class StreamingSFTDataset(IterableDataset):
    """Hugging Face datasets 流式指令数据读取。"""

    def __init__(self, source: DataSourceConfig, extras: SFTConfigExtras) -> None:
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
            normalized = _normalize_prompt_response(
                sample,
                prompt_key=self.extras.prompt_key,
                input_key=self.extras.input_key,
                response_key=self.extras.response_key,
                separator=self.extras.separator,
                prompt_prefix=self.extras.prompt_prefix,
                response_prefix=self.extras.response_prefix,
            )
            if not normalized:
                continue
            prompt, response = normalized
            yield {"prompt": prompt, "response": response, "source": self.source.name}
            counter += 1
            if self.source.max_samples is not None and counter >= self.source.max_samples:
                break


def _duplicate_with_weights(datasets: Sequence[Dataset[Any]], sources: Sequence[DataSourceConfig]) -> Dataset[Any]:
    min_weight = min(src.sampling_weight for src in sources)
    expanded: List[Dataset[Any]] = []
    for dataset, source in zip(datasets, sources):
        multiplier = max(1, int(round(source.sampling_weight / min_weight)))
        expanded.extend([dataset] * multiplier)
    return ConcatDataset(expanded) if len(expanded) > 1 else expanded[0]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class SFTDatasetModule(BaseDatasetModule):
    """指令微调数据模块（单轮或简单多轮可映射为 prompt/response）。"""

    def __init__(
        self,
        *,
        config: DataConfig,
        tokenizer,
        extras: SFTConfigExtras,
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
    ) -> "SFTDatasetModule":
        data_config = parse_data_config(cfg)
        sequence_length = int(cfg.get("sequence_length") or getattr(tokenizer, "model_max_length", 1024) or 1024)
        extras = SFTConfigExtras(
            sequence_length=sequence_length,
            prompt_key=str(cfg.get("prompt_key", "instruction")),
            input_key=cfg.get("input_key", "input"),
            response_key=str(cfg.get("response_key", "output")),
            separator=str(cfg.get("separator", "\n\n")),
            prompt_prefix=str(cfg.get("prompt_prefix", "")),
            response_prefix=str(cfg.get("response_prefix", "")),
            add_bos=bool(cfg.get("add_bos", True)),
            add_eos=bool(cfg.get("add_eos", True)),
            padding_strategy=str(cfg.get("padding", "max_length")),
        )
        return cls(config=data_config, tokenizer=tokenizer, extras=extras, seed=seed)

    def build_dataset(self) -> Dataset[Any]:
        datasets: List[Dataset[Any]] = []
        for source in self.config.sources:
            if source.type == "local":
                datasets.append(LocalSFTJsonlDataset(source, self.extras))
            elif source.type == "streaming":
                datasets.append(StreamingSFTDataset(source, self.extras))
            else:  # pragma: no cover
                raise ValueError(f"Unknown data source type {source.type}")
        if not datasets:
            raise ValueError("No datasets were constructed.")
        if len(datasets) == 1:
            return datasets[0]
        return _duplicate_with_weights(datasets, self.config.sources)

    def collate_fn(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        prompts = [str(ex["prompt"]) for ex in examples]
        responses = [str(ex["response"]) for ex in examples]
        source_names = [str(ex.get("source", "unknown")) for ex in examples]

        separator = self.extras.separator if self.extras.separator else "\n\n"
        texts = [f"{p}{separator}{r}" for p, r in zip(prompts, responses)]
        prompt_segments = [f"{p}{separator}" for p in prompts]

        padding = self._resolve_padding_strategy()
        encoded_full = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.sequence_length,
            truncation=True,
            padding=padding,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded_prompt = self.tokenizer(
            prompt_segments,
            add_special_tokens=True,
            max_length=self.sequence_length,
            truncation=True,
            padding="longest",
            return_attention_mask=False,
        )

        input_ids = encoded_full["input_ids"]
        attention_mask = encoded_full["attention_mask"]
        prompt_lengths = [len(ids) for ids in encoded_prompt["input_ids"]]

        if self.extras.add_bos and self.tokenizer.bos_token_id is not None:
            # 若 tokenizer 未自动添加 BOS，可在此处插入；当前依赖 add_special_tokens.
            pass
        if self.extras.add_eos and self.tokenizer.eos_token_id is not None:
            # 同理保证末尾包含 EOS
            pass

        labels = input_ids.clone()
        for idx, prompt_len in enumerate(prompt_lengths):
            clip_len = min(prompt_len, labels.shape[1])
            # prompt token 只提供条件信息，不应贡献监督信号；设置为 -100 让
            # CrossEntropyLoss 忽略这些位置，使 loss 仅聚焦在 response token。
            labels[idx, :clip_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": {
                "source": source_names,
                "prompt_length": prompt_lengths,
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


__all__ = ["SFTDatasetModule"]

# Notes:
#   - 适用于指令/单轮对话格式样本；多轮对话可在数据准备阶段拼接成单 prompt。
#   - 如需保留多字段 metadata，可扩展 Local/Streaming 数据集返回的字典结构。
#   - 训练脚本可通过 metadata['prompt_length'] 评估监督信号比重或做日志记录。
