from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """标准 LayerNorm 封装，保持 GPT-2 风格接口。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(hidden_states)


__all__ = ["LayerNorm"]
