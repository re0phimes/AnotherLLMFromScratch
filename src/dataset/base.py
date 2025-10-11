# === Dataset Framework Overview =================================================
# Entry Flow (handled by task-specific modules e.g. pretrain.py / sft.py):
#   1. Training script parses configs/train/*.yaml and obtains the `data` block.
#   2. Call `BaseDatasetModule.from_config(data_cfg, tokenizer=..., seed=...)` to
#      create a concrete Dataset module (subclass defined per task).
#   3. Subclass implements `build_dataset()` to construct torch Dataset and
#      `collate_fn()` to process tokenized samples into batches.
#   4. Trainer calls `module.build_dataloader(batch_size, ...)` to get a
#      ready-to-use PyTorch DataLoader for training/validation.
# Expected Sample Schema produced by subclasses (examples from pretraining):
#   {
#       "input_ids": Tensor[batch, seq_len],
#       "attention_mask": Tensor[batch, seq_len],
#       "labels": Tensor[batch, seq_len],
#       "metadata": Optional[dict] (e.g. source name, language tags)
#   }
# SFT/DPO tasks may adapt the schema (e.g. include separate prompt/response
# fields) but should remain consistent within each task-specific module.
# ===============================================================================
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import PreTrainedTokenizerBase
except ImportError:  # pragma: no cover - transformers 是必备依赖，提示更友好
    PreTrainedTokenizerBase = Any  # type: ignore[misc]


@dataclass
class DataSourceConfig:
    """单个数据源的配置描述。

    Attributes:
        name: 数据源名称，便于日志或混合策略中引用。
        type: 数据读取类型，local 读取本地文件，streaming 使用远程流式加载。
        path: 当 type=local 时指文件路径；type=streaming 时为数据集名称或远程路径。
        subset: 可选子集信息（如 Hugging Face datasets 的子配置）。
        sampling_weight: 混合时的权重，Trainer 会按比例重采样。
        max_samples: 可选样本上限，调试阶段常用。
        shuffle: 是否打乱样本读取顺序。
        streaming_kwargs: 额外的流式读取参数，如认证 token、batch 大小等。
    """

    name: str
    type: str
    path: str
    subset: Optional[str] = None
    sampling_weight: float = 1.0
    max_samples: Optional[int] = None
    shuffle: bool = True
    streaming_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.type not in {"local", "streaming"}:
            raise ValueError(f"Unsupported data source type '{self.type}'.")
        if self.sampling_weight <= 0:
            raise ValueError("sampling_weight must be positive.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive if specified.")


@dataclass
class DataConfig:
    """数据模块的综合配置。

    该结构由 YAML 配置解析而来，是所有 DatasetModule 的输入。
    """

    sources: Sequence[DataSourceConfig]
    tokenizer_batch_size: int = 2048
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = False

    def total_weight(self) -> float:
        return sum(source.sampling_weight for source in self.sources)

    def validate(self) -> None:
        if not self.sources:
            raise ValueError("At least one data source must be provided.")
        for source in self.sources:
            source.validate()
        if self.tokenizer_batch_size <= 0:
            raise ValueError("tokenizer_batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive.")


def _ensure_mapping(cfg: Mapping[str, Any]) -> MutableMapping[str, Any]:
    return dict(cfg)


def parse_data_config(cfg: Mapping[str, Any]) -> DataConfig:
    """将 YAML 中的 data 块解析为 DataConfig。

    兼容两种写法：
        1. 只有单一路径（path）字段。
        2. data_sources 列表，支持多数据源混合。
    """
    config_dict = _ensure_mapping(cfg)

    tokenizer_batch_size = int(config_dict.get("tokenizer_batch_size", 2048))
    num_workers = int(config_dict.get("num_workers", 4))
    prefetch_factor = int(config_dict.get("prefetch_factor", 2))
    pin_memory = bool(config_dict.get("pin_memory", True))
    drop_last = bool(config_dict.get("drop_last", False))

    default_type = str(config_dict.get("type", "local"))
    default_shuffle = bool(config_dict.get("shuffle", True))
    default_sampling_weight = float(config_dict.get("sampling_weight", 1.0))
    default_max_samples = config_dict.get("max_samples", None)
    default_subset = config_dict.get("subset")

    sources: list[DataSourceConfig] = []
    if "data_sources" in config_dict:
        for entry in config_dict["data_sources"]:
            entry_dict = _ensure_mapping(entry)
            source = DataSourceConfig(
                name=str(entry_dict.get("name") or entry_dict.get("path") or f"source_{len(sources)}"),
                type=str(entry_dict.get("type", default_type)),
                path=str(entry_dict["path"]),
                subset=entry_dict.get("subset", default_subset),
                sampling_weight=float(entry_dict.get("sampling_weight", default_sampling_weight)),
                max_samples=entry_dict.get("max_samples", entry_dict.get("limit", default_max_samples)),
                shuffle=bool(entry_dict.get("shuffle", default_shuffle)),
                streaming_kwargs=_ensure_mapping(entry_dict.get("streaming_kwargs", {})),
            )
            sources.append(source)
    else:
        if "path" not in config_dict:
            raise ValueError("data config must contain either 'path' or 'data_sources'.")
        source = DataSourceConfig(
            name=str(config_dict.get("name") or config_dict["path"]),
            type=default_type,
            path=str(config_dict["path"]),
            subset=default_subset,
            sampling_weight=default_sampling_weight,
            max_samples=default_max_samples,
            shuffle=default_shuffle,
            streaming_kwargs=_ensure_mapping(config_dict.get("streaming_kwargs", {})),
        )
        sources.append(source)

    data_config = DataConfig(
        sources=sources,
        tokenizer_batch_size=tokenizer_batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    data_config.validate()
    return data_config


class BaseDatasetModule(ABC):
    """Dataset 模块抽象基类。

    子类需实现构建底层 Dataset / IterableDataset 的逻辑，并通过 ``build_dataloader`` 暴露给 Trainer。
    """

    def __init__(
        self,
        *,
        config: DataConfig,
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.seed = seed

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        tokenizer: PreTrainedTokenizerBase,
        seed: int = 42,
    ) -> "BaseDatasetModule":
        data_config = parse_data_config(cfg)
        return cls(config=data_config, tokenizer=tokenizer, seed=seed)

    @abstractmethod
    def build_dataset(self) -> Dataset[Any]:
        """构建底层 torch Dataset 对象（Map-style 或 Iterable 均可）。"""

    @abstractmethod
    def collate_fn(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """定义批处理逻辑，负责将样本列表打包成张量 batch。"""

    def build_dataloader(
        self,
        *,
        batch_size: int,
        shuffle: Optional[bool] = None,
        drop_last: Optional[bool] = None,
    ) -> DataLoader[Any]:
        dataset = self.build_dataset()
        effective_shuffle = bool(self._infer_shuffle(dataset, shuffle))
        effective_drop_last = drop_last if drop_last is not None else self.config.drop_last

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=effective_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=effective_drop_last,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

    def _infer_shuffle(self, dataset: Dataset[Any], user_shuffle: Optional[bool]) -> bool:
        if user_shuffle is not None:
            return user_shuffle
        # IterableDataset 不支持 shuffle=True
        if isinstance(dataset, torch.utils.data.IterableDataset):
            return False
        # 默认若存在多个 source 且任何一个允许 shuffle，则启用
        if any(source.shuffle for source in self.config.sources):
            return True
        return False

    @staticmethod
    def compute_tokens_per_example(examples: Sequence[Mapping[str, Any]]) -> int:
        """简单估算 batch 中 token 数量，可供上层做 profiling。"""
        if not examples:
            return 0
        sample = examples[0]
        input_ids = sample.get("input_ids")
        if input_ids is None:
            return 0
        if isinstance(input_ids, torch.Tensor):
            return int(math.prod(input_ids.shape))
        if isinstance(input_ids, Sequence):
            return len(input_ids)
        return 0

# === Notes =====================================================================
# Subclass Implementation Checklist:
#   - Define __init__/from_config if additional parameters are required.
#   - Implement build_dataset(): read JSONL or streaming sources and apply
#     tokenization (often using config.tokenizer_batch_size for batching).
#   - Implement collate_fn(): pad/truncate tokens, produce tensors and optional
#     metadata (e.g., "source": str for downstream logging).
#   - Use BaseDatasetModule.build_dataloader to create train/val DataLoaders.
# Debugging Tips:
#   - Call compute_tokens_per_example on a small sample to estimate throughput.
#   - Log DataConfig.sources to confirm sampling weights and max_samples.
# ===============================================================================
