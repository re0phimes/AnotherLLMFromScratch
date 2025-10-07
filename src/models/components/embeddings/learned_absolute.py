from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LearnedAbsoluteEmbedding(nn.Module):
    """词嵌入 + 可学习绝对位置嵌入的组合模块。

    用于 GPT-2/BERT 等架构：每个位置 id (0, 1, ..., max_position-1) 都对应一个可训练向量。

    Args:
        vocab_size: 词表大小
        hidden_size: 嵌入维度（等于模型隐层维度）
        max_position_embeddings: 支持的最大序列长度
        dropout: 词嵌入与位置嵌入叠加后的 dropout 比例
        pad_token_id: 选填；若提供，将在根据 attention_mask 计算 position_ids 时忽略 padding 位置
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        *,
        dropout: float = 0.0,
        pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive.")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 缓存 [0, 1, ..., max_position_embeddings-1]
        self.register_buffer(
            "_position_ids",
            torch.arange(max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token id 序列
            position_ids: (B, T) 可选；若为空则自动生成
            attention_mask: (B, T) 可选；用来在存在 padding 时计算位置 id
        Returns:
            embeddings: (B, T, hidden_size)
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq_length).")

        batch_size, seq_length = input_ids.size()

        if position_ids is None:
            position_ids = self._create_position_ids(input_ids, attention_mask)
        else:
            if position_ids.shape != (batch_size, seq_length):
                raise ValueError("position_ids must have the same shape as input_ids.")

        if position_ids.max().item() >= self.max_position_embeddings:
            raise ValueError(
                f"position_ids contains index >= max_position_embeddings ({self.max_position_embeddings})."
            )

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeds
        embeddings = self.dropout(embeddings)
        return embeddings

    def _create_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return self._position_ids[:, : input_ids.size(1)].repeat(input_ids.size(0), 1)

        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")

        if self.pad_token_id is None:
            # 不考虑 padding，按因果顺序生成
            cumsum = attention_mask.long().cumsum(dim=1) - 1
            cumsum = cumsum.clamp(min=0)
            return cumsum

        # 兼容 padding：pad_token 对应的 attention_mask 应为 0
        mask = (input_ids != self.pad_token_id).long() * attention_mask.long()
        cumsum = mask.cumsum(dim=1) - 1
        cumsum = cumsum.masked_fill(mask == 0, 0)
        return cumsum


__all__ = ["LearnedAbsoluteEmbedding"]
