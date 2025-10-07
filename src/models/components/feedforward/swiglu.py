from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardSwiGLU(nn.Module):
    """SwiGLU 前馈网络，用于 Qwen2 等现代架构。"""

    def __init__(
        self,
        hidden_size: int,
        *,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * ffn_multiplier)
        if intermediate_size % 2 != 0:
            intermediate_size += 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = nn.Dropout(dropout)

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(hidden_states)
        x2 = self.w2(hidden_states)
        x = F.silu(x1) * x2
        x = self.w3(x)
        x = self.dropout(x)
        return x


__all__ = ["FeedForwardSwiGLU"]
