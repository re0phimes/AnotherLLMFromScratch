from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardGELU(nn.Module):
    """标准前馈网络 (GELU 激活)，用于 GPT-2 like 架构。"""

    def __init__(
        self,
        hidden_size: int,
        *,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
        activation: Literal["gelu", "gelu_new"] = "gelu",
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * ffn_multiplier)
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=bias)
        if activation not in {"gelu", "gelu_new"}:
            raise ValueError("activation must be 'gelu' or 'gelu_new'.")
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(hidden_states)
        if self.activation == "gelu_new":
            x = F.gelu(x, approximate="tanh")
        else:
            x = F.gelu(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


__all__ = ["FeedForwardGELU"]
