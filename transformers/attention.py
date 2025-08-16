"""
Implements the attention layer in https://arxiv.org/abs/1706.03762
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(
        self, input_size: int, key_size: int, value_size: int,
    ) -> None:
        super().__init__()
        self.key_size = key_size
        self.query_size = key_size
        self.value_size = value_size
        self.scale_value = torch.sqrt(
            torch.tensor(key_size, dtype=torch.float)
        )

        self.Q = nn.Linear(input_size, key_size, bias=False)
        self.K = nn.Linear(input_size, key_size, bias=False)
        self.V = nn.Linear(input_size, value_size, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mask is -1 where values should be masked
        """
        xQ, xK, xV = self.Q(q), self.K(k), self.V(v)
        attention = (xQ @ xK.transpose(-2, -1)) / self.scale_value
        if mask is not None:
            attention = attention.masked_fill(mask == -1, -1e9)

        attention = torch.softmax((attention), dim=-1)
        return attention @ xV, attention


class MultiAttentionHead(nn.Module):
    def __init__(
        self,
        input_size: int,
        key_size: int,
        value_size: int,
        projection_size: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        """
        `projection_size` is often the same as the model size which is
        specified by the input_size. However added as a separate parameter for
        flexibility
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(input_size, key_size, value_size)
                for _ in range(n_heads)
            ]
        )
        self.W = nn.Linear(value_size * n_heads, projection_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Recall head returns a tuple of att * value and attentions
        attentions = [head(q, k, v, mask) for head in self.heads]
        # Concatenate along the final dimension to get
        # (Batch, Seq_len, value_size * n_heads)
        outputs = [value for value, _ in attentions]
        outputs = torch.cat(outputs, dim=-1)
        return self.dropout(self.W(outputs))
