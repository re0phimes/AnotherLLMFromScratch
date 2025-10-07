from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class Qwen2ModelOutput:
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[Optional[torch.Tensor], ...]] = None


class Qwen2Model(nn.Module):
    """Qwen2-like causal LM backbone (placeholder)."""

    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Qwen2Model is not yet implemented.")


__all__ = ["Qwen2Model", "Qwen2ModelOutput"]


