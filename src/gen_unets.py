from typing import Union
from src.embedders import SumEmbedder, CrossAttentionEmbedder
from src.pos_enc import MultiWeightedPositionalEncoding, MultiRotationalEncoding
from src.transformer_blocks import TransformerBlock

from torch import nn


class FlatMaskedUNeTr(nn.Module):
    def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            output_dim: int,
            kernel_size: int,
            input_channels: int,
            roi: int,
            num_patches: int,
            n_dim: int = 2,
            num_stages: int = 4,
            rotaional_encoding: bool = False,
            cross_attn_embedder: bool = False,
            num_encodings: int = 24
    ):
        super(FlatMaskedUNeTr, self).__init__()
        self.encoder = nn.ModuleList([TransformerBlock(latent_dim) for _ in range(num_stages)])
        self.decoder = nn.ModuleList([TransformerBlock(latent_dim) for _ in range(num_stages)])
        self.fc_in = nn.Linear(input_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, output_dim)
        if rotaional_encoding and n_dim > 1:
            encoding = MultiRotationalEncoding(roi, latent_dim, num_encodings, n_dim==3)
        else:
            encoding = MultiWeightedPositionalEncoding(num_patches, latent_dim, num_encodings)
        if cross_attn_embedder:
            self.positional_embedding = CrossAttentionEmbedder(
                encoding, latent_dim, num_encodings
            )
        else:
            self.positional_embedding = SumEmbedder(
                encoding
            )

    def forward(self, x, masks=None, selected_indices=None):
        x = self.fc_in(x)
        x = self.positional_embedding(x, selected_indices)
        skips = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skips.append(x)

        for i, decoder_block in enumerate(self.decoder):
            mask = masks[i] if masks is not None else None
            x = decoder_block(x, mask=mask)
            x += skips[-(i+1)]

        x = self.fc_out(x)
        return x
