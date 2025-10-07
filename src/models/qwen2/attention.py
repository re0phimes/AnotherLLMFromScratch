from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..components import (
    GroupedQueryAttention,
    GroupedQueryAttentionCache,
    GroupedQueryAttentionOutput,
)


class Qwen2Attention(nn.Module):
    """Qwen2 专用注意力包装，基于 GQA + RoPE + KV cache。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        rope_dim: Optional[int] = None,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        qkv_bias: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_dim=rope_dim,
            rope_base=rope_base,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[GroupedQueryAttentionCache] = None,
        use_cache: bool = False,
        need_weights: bool = False,
    ) -> GroupedQueryAttentionOutput:
        return self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            need_weights=need_weights,
        )


__all__ = ["Qwen2Attention"]


