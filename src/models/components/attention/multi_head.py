from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiHeadAttentionOutput:
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class MultiHeadSelfAttention(nn.Module):
    """
    标准多头自注意力 (GPT-2 like)。

    记号:
      - B: batch size
      - T: 序列长度
      - H: 头数
      - Dh: 每头维度 (head_dim)
      - C: 总隐维 (C = H * Dh)
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
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_dropout = float(attn_dropout)
        self.resid_dropout = float(resid_dropout)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.resid_drop = nn.Dropout(self.resid_dropout)
        self.use_flash = bool(use_flash and hasattr(F, "scaled_dot_product_attention"))
        self._reset_parameters()

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """将输入的 attention_mask 广播到 Flash/SDPA 期望的 4 维形状。"""

        mask = attention_mask.to(device=attention_mask.device)

        if mask.dim() == 2:  # (B, S)
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:  # (B, T, S)
            mask = mask[:, None, :, :]
        elif mask.dim() != 4:
            raise ValueError(
                "attention_mask must have 2, 3 or 4 dimensions, got " f"{mask.dim()}"
            )

        if mask.size(0) != batch_size:
            if mask.size(0) == 1:
                mask = mask.expand(batch_size, *mask.shape[1:])
            else:
                raise ValueError(
                    "attention_mask batch dimension mismatch: "
                    f"expected {batch_size}, got {mask.size(0)}"
                )
        if mask.size(-1) != src_len:
            raise ValueError(
                "attention_mask last dimension must match source length: "
                f"expected {src_len}, got {mask.size(-1)}"
            )
        if mask.size(-2) not in (1, tgt_len):
            raise ValueError(
                "attention_mask penultimate dimension must broadcast to tgt_len: "
                f"expected 1 or {tgt_len}, got {mask.size(-2)}"
            )

        if mask.dtype == torch.bool:
            return mask

        if mask.is_floating_point():
            # 判定是否是二值掩码（0/1），是的话转为 bool；否则按 additive mask 处理。
            if torch.all((mask == 0) | (mask == 1)):
                return mask.to(dtype=torch.bool)
            return mask.to(dtype=dtype)

        if mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            if torch.all((mask == 0) | (mask == 1)):
                return mask.to(dtype=torch.bool)
            return mask.to(dtype=dtype)

        raise TypeError(f"Unsupported attention_mask dtype: {mask.dtype}")

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        is_causal: bool = True,
        need_weights: bool = False,
    ) -> MultiHeadAttentionOutput:
        B, T, C = hidden_states.shape
        if C != self.embed_dim:
            raise ValueError(f"hidden_states last dim ({C}) != embed_dim ({self.embed_dim}).")

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        prepared_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            prepared_mask = self._prepare_attention_mask(
                attention_mask,
                batch_size=B,
                tgt_len=T,
                src_len=k.size(-2),
                dtype=q.dtype,
            )

        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=prepared_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=is_causal and (attention_mask is None),
            )
            attn_weights = None
        else:
            scale = 1.0 / (self.head_dim ** 0.5)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if prepared_mask is not None:
                if prepared_mask.dtype == torch.bool:
                    attn_scores = attn_scores.masked_fill(~prepared_mask, float("-inf"))
                else:
                    attn_scores = attn_scores + prepared_mask
            elif is_causal:
                causal_mask = torch.ones((T, T), dtype=torch.bool, device=hidden_states.device).tril()
                attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_drop(attn_probs)
            attn_output = torch.matmul(attn_probs, v)
            attn_weights = attn_probs if need_weights else None

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_drop(attn_output)

        if need_weights and attn_weights is not None:
            avg_weights = attn_weights.mean(dim=1)
        else:
            avg_weights = None

        return MultiHeadAttentionOutput(hidden_states=attn_output, attention_weights=avg_weights)


__all__ = ["MultiHeadSelfAttention", "MultiHeadAttentionOutput"]


