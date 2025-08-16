from typing import Optional

import torch
import torch.nn as nn
from transformer_blocks import PositionalEmbedder, Encoder, Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        max_sequence_length: int,
        learn_pos_embeddings: bool,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        key_size: int,
        value_size: int,
        n_attention_heads: int,
        ff_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=0
        )
        self.positional_embedder = PositionalEmbedder(
            embedding_size, max_sequence_length, train=learn_pos_embeddings
        )
        self.embedding_scale_factor = torch.sqrt(
            torch.tensor(embedding_size, dtype=torch.float)
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            embedding_size,
            ff_size,
            key_size,
            value_size,
            n_attention_heads,
            n_encoder_blocks,
        )

        self.decoder = Decoder(
            embedding_size,
            ff_size,
            key_size,
            value_size,
            n_attention_heads,
            n_decoder_blocks,
        )

        self.linear = nn.Linear(embedding_size, vocab_size)

    def _embed(self, x: torch.LongTensor) -> torch.Tensor:
        x = self.vocab_embedding(x) * self.embedding_scale_factor
        x = self.positional_embedder(x)
        return self.embedding_dropout(x)

    def forward(
        self,
        encoder_inputs: torch.LongTensor,
        decoder_inputs: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Input is long tensor which is converted to float by the embedding
        layer
        """
        encoder_embs = self._embed(encoder_inputs)
        encoder_output = self.encoder(encoder_embs)

        decoder_embs = self._embed(decoder_inputs)
        decoder_output = self.decoder(decoder_embs, encoder_output, mask)
        return self.linear(decoder_output)
