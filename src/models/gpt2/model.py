from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import (
    FeedForwardGELU,
    LayerNorm,
    LearnedAbsoluteEmbedding,
    MultiHeadAttentionOutput,
    MultiHeadSelfAttention,
)


@dataclass
class GPT2ModelOutput:
    """Causal LM forward输出结构。"""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[Optional[torch.Tensor], ...]] = None


class GPT2Block(nn.Module):
    """GPT-2 Transformer Block (Pre-LayerNorm)。

    结构: LN -> Multi-Head Attention -> Residual -> LN -> MLP(GELU) -> Residual

    Args:
        embed_dim: 隐藏维度 (C)
        num_heads: 注意力头数 (H)
        mlp_multiplier: 前馈层扩大倍率 (默认4x)
        attn_dropout: 注意力权重 dropout
        resid_dropout: Residual dropout (应用于注意力和 MLP 输出)
        activation: MLP激活函数 ('gelu' 或 'gelu_new')
        layer_norm_eps: LayerNorm epsilon
        qkv_bias: 是否为 QKV 使用 bias
        use_flash: 是否优先使用 PyTorch SDPA / Flash Attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_multiplier: float = 4.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        qkv_bias: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.ln_2 = LayerNorm(embed_dim, eps=layer_norm_eps)
        self.mlp = FeedForwardGELU(
            hidden_size=embed_dim,
            ffn_multiplier=mlp_multiplier,
            dropout=resid_dropout,
            activation="gelu_new" if activation == "gelu_new" else "gelu",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """执行单个 Transformer block 前向。

        Args:
            hidden_states: 输入张量，形状 (B, T, C)
            attention_mask: 注意力掩码，可广播到 (B, H, T, S)
            need_weights: 是否返回平均注意力权重

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (更新后的隐藏状态, 注意力权重或 None)
        """

        attn_norm = self.ln_1(hidden_states)
        attn_output: MultiHeadAttentionOutput = self.attn(
            attn_norm,
            attention_mask=attention_mask,
            is_causal=True,
            need_weights=need_weights,
        )
        hidden_states = hidden_states + attn_output.hidden_states

        mlp_norm = self.ln_2(hidden_states)
        mlp_output = self.mlp(mlp_norm)
        hidden_states = hidden_states + mlp_output

        return hidden_states, attn_output.attention_weights


class GPT2Model(nn.Module):
    """GPT-2 风格的 Causal LM 主干网络。

    组件组成:
        - LearnedAbsoluteEmbedding (词 + 绝对位置嵌入)
        - 多层 GPT2Block 堆叠 (Pre-LN)
        - 输出 LayerNorm + tied LM Head

    Args:
        vocab_size: 词表大小
        n_layer: Transformer block 数量
        n_head: 注意力头数
        n_embd: 隐藏维度 (embed dim)
        block_size: 最大上下文长度 (position embedding 长度)
        attn_dropout/resid_dropout: 注意力与 residual dropout
        mlp_multiplier: 前馈层扩展倍率
        layer_norm_eps: LayerNorm epsilon
        qkv_bias: 是否为 QKV 投影使用 bias
        use_flash: 是否优先使用 PyTorch SDPA (Flash Attention)
        pad_token_id: 可选 padding token id (用于生成 position_ids)
        output_hidden_states: 是否在输出中附带所有 hidden states
        output_attentions: 是否在输出中附带所有 attention maps
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_multiplier: float = 4.0,
        layer_norm_eps: float = 1e-5,
        qkv_bias: bool = True,
        use_flash: bool = True,
        pad_token_id: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> None:
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "block_size": block_size,
        }
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions

        self.embeddings = LearnedAbsoluteEmbedding(
            vocab_size=vocab_size,
            hidden_size=n_embd,
            max_position_embeddings=block_size,
            dropout=resid_dropout,
            pad_token_id=pad_token_id,
        )

        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    embed_dim=n_embd,
                    num_heads=n_head,
                    mlp_multiplier=mlp_multiplier,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    layer_norm_eps=layer_norm_eps,
                    qkv_bias=qkv_bias,
                    use_flash=use_flash,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd, eps=layer_norm_eps)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> GPT2ModelOutput:
        """执行 GPT-2 前向推理/训练。

        Args:
            input_ids: Token ids，形状 (B, T)
            attention_mask: 掩码 (B, T)，1 表示可见，0 表示被遮挡
            position_ids: 可选位置索引 (B, T)，默认自增
            need_weights: 是否返回注意力权重
            labels: 可选预测标签 (B, T)，设置后计算自回归 loss

        Returns:
            GPT2ModelOutput: 包含 logits、loss(如果 labels 提供)、隐藏状态和注意力权重
        """
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        all_hidden_states = [] if self.output_hidden_states else None
        all_attentions = [] if (need_weights or self.output_attentions) else None

        for block in self.blocks:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = block(
                hidden_states,
                attention_mask=attention_mask,
                need_weights=need_weights or self.output_attentions,
            )
            if all_attentions is not None:
                all_attentions.append(attn_weights)

        hidden_states = self.ln_f(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids.")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return GPT2ModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )


__all__ = ["GPT2Model", "GPT2ModelOutput", "GPT2Block"]
