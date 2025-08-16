"""
Transformer building blocks
"""
from typing import Optional

import torch
import torch.nn as nn

from attention import MultiAttentionHead


class FFN(nn.Module):
    def __init__(
        self, input_size: int, inner_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.f1 = nn.Linear(input_size, inner_size)
        self.f2 = nn.Linear(inner_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.f2(torch.relu(self.f1(x))))


class PositionalEmbedder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        max_sequence_length: int,
        scaling_factor: float = 10000.0,
        train: bool = False,
    ) -> torch.Tensor:
        super().__init__()

        def _get_angles(
            scaling_factor: float, embedding_size: int
        ) -> torch.Tensor:
            dimension = torch.arange(embedding_size, dtype=torch.float)
            exponent = (2 * dimension) / embedding_size
            denominator = scaling_factor ** exponent
            return 1.0 / denominator

        embedding_size = embedding_size
        angles = _get_angles(scaling_factor, embedding_size)
        angles = torch.arange(max_sequence_length).reshape(-1, 1) * angles
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])
        self.positional_embeddings = nn.Parameter(angles, requires_grad=train)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_embeddings


class EncoderBlock(nn.Module):
    """
    Self-attention and feed-forward module
    """

    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
    ):
        super().__init__()
        self.attention_heads = MultiAttentionHead(
            input_size, key_size, value_size, input_size, n_heads
        )
        self.ffn = FFN(input_size, ff_size)

        # Normalise over last dimension only
        self.ln1 = nn.LayerNorm((input_size))
        self.ln2 = nn.LayerNorm((input_size))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input gets multiplied three times to q, k, v
        x = self.ln1(self.attention_heads(x, x, x) + x)
        x = self.ln2(self.ffn(x) + x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
        n_layers: int,
    ):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(
                    input_size, ff_size, key_size, value_size, n_heads,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_blocks(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.mah1 = MultiAttentionHead(
            input_size, key_size, value_size, input_size, n_heads
        )

        self.mah2 = MultiAttentionHead(
            input_size, key_size, value_size, input_size, n_heads
        )

        self.ffn = FFN(input_size, ff_size)
        self.ln1 = nn.LayerNorm((input_size))
        self.ln2 = nn.LayerNorm((input_size))
        self.ln3 = nn.LayerNorm((input_size))

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.ln1(self.mah1(x, x, x, mask) + x)
        out = self.ln2(self.mah2(out, encoder_output, encoder_output) + out)
        out = self.ln3(self.ffn(out) + out)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        ff_size: int,
        key_size: int,
        value_size: int,
        n_heads: int,
        n_layers: int,
    ) -> None:

        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    input_size, ff_size, key_size, value_size, n_heads,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for decoder in self.decoder_blocks:
            x = decoder(x, encoder_output, mask)
        return x
