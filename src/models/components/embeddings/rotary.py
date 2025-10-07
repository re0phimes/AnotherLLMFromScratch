from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RotaryCache:
    cos: torch.Tensor
    sin: torch.Tensor


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE) 实现，支持缓存与自定义 position_ids。

    记号:
      - B: batch size
      - T: 序列长度
      - D: 每头应用 RoPE 的维度 (需偶数)
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        self.dim = dim
        self.base = float(base)
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
        self._seq_len_cached = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._seq_len_cached and self._cos_cached.numel() > 0:
            return
        half_dim = self.dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,f->tf", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
        self._seq_len_cached = seq_len

    def get_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        position_ids: Optional[torch.Tensor] = None,
    ) -> RotaryCache:
        self._build_cache(seq_len, device, dtype)
        if position_ids is None:
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        else:
            cos = self._cos_cached.index_select(0, position_ids.reshape(-1)).view(*position_ids.shape, -1)
            sin = self._sin_cached.index_select(0, position_ids.reshape(-1)).view(*position_ids.shape, -1)
        return RotaryCache(cos=cos, sin=sin)

    @staticmethod
    def apply_rotary(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        rotary_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even.")
        if rotary_dim > q.size(-1) or rotary_dim > k.size(-1):
            raise ValueError("rotary_dim exceeds head dimension.")

        q1, q2 = q[..., :rotary_dim], q[..., rotary_dim:]
        k1, k2 = k[..., :rotary_dim], k[..., rotary_dim:]

        cos = RotaryEmbedding._match_shape(cos, q1)
        sin = RotaryEmbedding._match_shape(sin, q1)

        q_rot = (q1 * cos) + (RotaryEmbedding._rotate_half(q1) * sin)
        k_rot = (k1 * cos) + (RotaryEmbedding._rotate_half(k1) * sin)

        q = torch.cat([q_rot, q2], dim=-1)
        k = torch.cat([k_rot, k2], dim=-1)
        return q, k

    @staticmethod
    def _match_shape(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.dim() == target.dim() - 1:
            x = x.unsqueeze(0)
        while x.dim() < target.dim():
            x = x.unsqueeze(0)
        return x

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.view(*x.shape[:-1], -1, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


__all__ = ["RotaryEmbedding", "RotaryCache"]


