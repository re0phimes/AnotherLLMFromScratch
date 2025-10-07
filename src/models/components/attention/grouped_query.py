from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings.rotary import RotaryEmbedding


@dataclass
class GroupedQueryAttentionCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class GroupedQueryAttentionOutput:
    hidden_states: torch.Tensor
    cache: Optional[GroupedQueryAttentionCache] = None
    attention_weights: Optional[torch.Tensor] = None


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention)，支持 RoPE 与 KV Cache。

    记号:
      - B: batch size
      - T: 当前输入序列长度
      - H: 查询头数 (num_heads)
      - Hk: KV 头数 (num_kv_heads)
      - Dh: 每头维度 (head_dim)
      - C: 总隐维 (C = H * Dh)
    """

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
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped query attention.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_repeat = num_heads // num_kv_heads

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * kv_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = float(attn_dropout)
        self.resid_dropout = float(resid_dropout)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.resid_drop = nn.Dropout(self.resid_dropout)
        self.use_flash = bool(use_flash and hasattr(F, "scaled_dot_product_attention"))

        self.rotary_dim = rope_dim if rope_dim is not None else self.head_dim
        if self.rotary_dim > self.head_dim:
            raise ValueError("rope_dim cannot exceed head_dim.")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rope_dim must be even.")
        self.rotary = RotaryEmbedding(self.rotary_dim, base=rope_base)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)

        nn.init.xavier_uniform_(self.kv_proj.weight)
        if self.kv_proj.bias is not None:
            nn.init.zeros_(self.kv_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.kv_repeat == 1:
            return k, v
        k = k.unsqueeze(2).repeat(1, 1, self.kv_repeat, 1, 1)
        v = v.unsqueeze(2).repeat(1, 1, self.kv_repeat, 1, 1)
        B, Hk, repeat, T, Dh = k.shape
        k = k.reshape(B, Hk * repeat, T, Dh)
        v = v.reshape(B, Hk * repeat, T, Dh)
        return k, v

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
        B, T, C = hidden_states.shape
        if C != self.embed_dim:
            raise ValueError(f"hidden_states last dim ({C}) != embed_dim ({self.embed_dim}).")

        q = self.q_proj(hidden_states)
        kv = self.kv_proj(hidden_states)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hk, T, Dh)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value.key.size(-2) if past_key_value is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        rotary_cache = self.rotary.get_cache(
            seq_len=int(position_ids.max().item()) + 1,
            device=hidden_states.device,
            dtype=q.dtype,
            position_ids=position_ids,
        )
        cos = rotary_cache.cos.unsqueeze(1)  # (B, 1, T, D)
        sin = rotary_cache.sin.unsqueeze(1)
        q, k = RotaryEmbedding.apply_rotary(q, k, cos, sin, rotary_dim=self.rotary_dim)

        if past_key_value is not None:
            k = torch.cat([past_key_value.key, k], dim=2)
            v = torch.cat([past_key_value.value, v], dim=2)

        new_cache = GroupedQueryAttentionCache(key=k, value=v) if use_cache else None

        k_expanded, v_expanded = self._expand_kv(k, v)

        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q,
                k_expanded,
                v_expanded,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
            attn_weights = None
        else:
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    attn_scores = attn_scores.masked_fill(~attention_mask, float("-inf"))
                else:
                    attn_scores = attn_scores + attention_mask
            else:
                causal_mask = torch.ones((q.size(-2), k_expanded.size(-2)), dtype=torch.bool, device=q.device).tril()
                attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_drop(attn_probs)
            attn_output = torch.matmul(attn_probs, v_expanded)
            attn_weights = attn_probs if need_weights else None

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_drop(attn_output)

        if need_weights and attn_weights is not None:
            avg_weights = attn_weights.mean(dim=1)
        else:
            avg_weights = None

        return GroupedQueryAttentionOutput(
            hidden_states=attn_output,
            cache=new_cache,
            attention_weights=avg_weights,
        )


__all__ = [
    "GroupedQueryAttention",
    "GroupedQueryAttentionOutput",
    "GroupedQueryAttentionCache",
]


