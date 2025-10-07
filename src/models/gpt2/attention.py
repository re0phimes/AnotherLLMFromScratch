from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..components import MultiHeadSelfAttention, MultiHeadAttentionOutput


class GPT2Attention(nn.Module):
    """GPT-2 专用自注意力包装。

    直接复用共享的 MultiHeadSelfAttention，统一 GPT-2 模块内部接口。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        qkv_bias: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
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
        need_weights: bool = False,
    ) -> MultiHeadAttentionOutput:
        return self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
            need_weights=need_weights,
        )


__all__ = ["GPT2Attention"]
