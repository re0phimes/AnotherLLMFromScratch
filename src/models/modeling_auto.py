from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Tuple, Type, Union

import torch.nn as nn

from .gpt2.config import GPT2Config
from .gpt2.model import GPT2Model

ModelConfig = Union[GPT2Config]
ModelClass = Type[nn.Module]

_MODEL_REGISTRY: Dict[str, Tuple[Type[ModelConfig], ModelClass]] = {
    "gpt2": (GPT2Config, GPT2Model),
    # 预留扩展位："qwen2": (Qwen2Config, Qwen2Model),
}


def _resolve_model_family(model_family: str) -> Tuple[Type[ModelConfig], ModelClass]:
    family = model_family.lower()
    if family not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. Available: {list(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[family]


class AutoConfig:
    """根据配置字典自动实例化对应模型族的配置对象。"""

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> ModelConfig:
        if "model_family" not in config_dict and "model_type" not in config_dict:
            raise ValueError("config must provide 'model_family' (or legacy 'model_type').")
        family = str(config_dict.get("model_family") or config_dict.get("model_type"))
        config_cls, _ = _resolve_model_family(family)
        config_payload = dict(config_dict)
        config_payload.pop("model_family", None)
        config_payload.pop("model_type", None)
        return config_cls.from_dict(config_payload)

    @classmethod
    def to_dict(cls, config: ModelConfig) -> Dict[str, Any]:
        if not is_dataclass(config):
            raise TypeError("config must be a dataclass instance.")
        result = asdict(config)
        for family, (config_cls, _) in _MODEL_REGISTRY.items():
            if isinstance(config, config_cls):
                result["model_family"] = family
                break
        else:
            raise ValueError("Unregistered config type: cannot export to dict.")
        return result


class AutoModelForCausalLM:
    """根据配置自动构建 Causal LM 模型实例。"""

    @classmethod
    def from_config(cls, config: ModelConfig) -> nn.Module:
        for _, (config_cls, model_cls) in _MODEL_REGISTRY.items():
            if isinstance(config, config_cls):
                return model_cls(**config.to_model_kwargs())
        raise ValueError("Unsupported config type. Did you forget to register it in _MODEL_REGISTRY?")

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> nn.Module:
        config = AutoConfig.from_dict(config_dict)
        return cls.from_config(config)


__all__ = ["AutoConfig", "AutoModelForCausalLM"]

