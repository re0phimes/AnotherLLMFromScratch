from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


_VALID_ACTIVATIONS = {"gelu", "gelu_new"}


@dataclass
class GPT2Config:
    """GPT-2 模型配置定义。

    对应架构文档要求：必须提供 vocab_size, n_layer, n_head, n_embd, block_size 等字段。
    额外参数用于控制 dropout、激活函数等，可通过 YAML 配置覆盖。
    """

    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int

    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    mlp_multiplier: float = 4.0
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    qkv_bias: bool = True
    use_flash: bool = True
    pad_token_id: Optional[int] = None
    output_hidden_states: bool = False
    output_attentions: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive.")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive.")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head to form head_dim.")
        if self.activation not in _VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {_VALID_ACTIVATIONS}.")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_model_kwargs(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "attn_dropout": self.attn_dropout,
            "resid_dropout": self.resid_dropout,
            "mlp_multiplier": self.mlp_multiplier,
            "layer_norm_eps": self.layer_norm_eps,
            "qkv_bias": self.qkv_bias,
            "use_flash": self.use_flash,
            "pad_token_id": self.pad_token_id,
            "output_hidden_states": self.output_hidden_states,
            "output_attentions": self.output_attentions,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "block_size": self.block_size,
            "attn_dropout": self.attn_dropout,
            "resid_dropout": self.resid_dropout,
            "mlp_multiplier": self.mlp_multiplier,
            "activation": self.activation,
            "layer_norm_eps": self.layer_norm_eps,
            "qkv_bias": self.qkv_bias,
            "use_flash": self.use_flash,
            "pad_token_id": self.pad_token_id,
            "output_hidden_states": self.output_hidden_states,
            "output_attentions": self.output_attentions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPT2Config":
        return cls(**data)


__all__ = ["GPT2Config"]


