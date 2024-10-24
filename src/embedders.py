import torch
import torch.nn as nn
from src.pos_enc import MultipleEncoding
from src.gating_blocks import BernoulliGating
from logging import Logger as log
from typing import Union


class Embedder(nn.Module):
    """
    Base class for applying an embedding and optional gating mechanism to input data.

    Args:
        embedding (nn.Module): A module that provides positional or other types of embedding for the input.
        gating (Union[nn.Module, None], optional): Optional gating module to apply after embedding. Defaults to None.
        num_patches (Union[int, None], optional): Number of patches to expect in the input. Required if using `BernoulliGating`. Defaults to None.
        num_channels (Union[int, None], optional): Number of channels in the input. Required if using `BernoulliGating`. Defaults to None.
        **kwargs: Additional arguments for flexibility.

    Attributes:
        embedding (nn.Module): The embedding module applied to the input.
        gating (Union[nn.Module, None]): Optional gating module for controlling information flow between multiple encodings.
    """

    def __init__(
            self, 
            embedding: nn.Module, 
            gating: Union[nn.Module, None] = None, 
            num_patches: Union[int, None] = None, 
            num_channels : Union[int, None] = None, 
            **kwargs
        ) -> None:
        super().__init__()
        self.embedding = embedding
        self.multi_enc = False
        if embedding is MultipleEncoding:
            self.multi_enc = True
            if embedding.num_encodings > 1:
                if gating is None:
                    log.warning(
                        "No gating given on multiple encoding!\nUsing \"BernoulliGating!\""
                    )
                    gating = BernoulliGating(num_patches, num_channels, embedding.num_encodings)
        self.gating = gating

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("This module is only abstract!")


class SumEmbedder(Embedder):
    """
    Sum-based embedder that adds the input tensor to the generated embeddings.

    Args:
        embedding (nn.Module): A module that provides positional or other types of embedding for the input.
        gating (Union[nn.Module, None], optional): Optional gating module to apply after embedding. Defaults to None.
        num_patches (Union[int, None], optional): Number of patches to expect in the input. Required if using `BernoulliGating`. Defaults to None.
        num_channels (Union[int, None], optional): Number of channels in the input. Required if using `BernoulliGating`. Defaults to None.
    """

    def __init__(
            self, 
            encoder: nn.Module, 
            gating: Union[nn.Module, None] = None, 
            num_patches: Union[int, None] = None, 
            num_channels : Union[int, None] = None
        ) -> None:
        super().__init__(
            encoder,
            gating,
            num_patches,
            num_channels
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for SumEmbedder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_patches).

        Returns:
            torch.Tensor: The result of adding the input tensor to the embedding tensor.
        """
        pe = self.embedding(x)
        if self.gating is not None:
            pe = self.gating(pe)
        if self.multi_enc:
            pe = pe.sum(1)
        pe = pe.squeeze(1)
        if len(pe.shape) > 3:
            pe = pe.squeeze(1)
        return x + pe.permute(0, 2, 1)[:, :, :, None, None]


class CrossAttentionEmbedder(Embedder):
    """
    Cross-attention based embedder that applies multi-head attention between the input and positional embeddings.

    Args:
        embedding (nn.Module): A module that provides positional or other types of embedding for the input.
        embed_dim (int): Dimension for the attention embedding.
        num_heads (int): Number of heads for the Multi-head Attention.
        gating (Union[nn.Module, None], optional): Optional gating module to apply after embedding. Defaults to None.
        num_patches (Union[int, None], optional): Number of patches to expect in the input. Required if using `BernoulliGating`. Defaults to None.
        num_channels (Union[int, None], optional): Number of channels in the input. Required if using `BernoulliGating`. Defaults to None.

    Attributes:
        embedding (nn.Module): The embedding module applied to the input.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        embed_dim (int): Dimensionality of the embedding space.
        attention (nn.MultiheadAttention): Multi-head attention layer for performing cross-attention.
    """

    def __init__(
            self,
            embedding: nn.Module, 
            embed_dim: int, 
            num_heads: int,
            gating: Union[nn.Module, None] = None, 
            num_patches: Union[int, None] = None, 
            num_channels : Union[int, None] = None
        ) -> None:
        """
        Initializes the CrossAttentionEmbedder.

        Args:
            embedding (nn.Module): A module that provides positional or other types of embedding for the input.
            embed_dim (int): Dimension for the attention embedding.
            num_heads (int): Number of heads for the Multi-head Attention.
            gating (Union[nn.Module, None], optional): Optional gating module to apply after embedding. Defaults to None.
            num_patches (Union[int, None], optional): Number of patches to expect in the input. Required if using `BernoulliGating`. Defaults to None.
            num_channels (Union[int, None], optional): Number of channels in the input. Required if using `BernoulliGating`. Defaults to None.
        """
        super().__init__(
            embedding,
            gating,
            num_patches,
            num_channels
        )
        self.embedding = embedding
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Multi-head attention for cross-attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for CrossAttentionEmbedder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width) or 
                              (batch_size, num_channels, depth, height, width) for 2D or 3D data respectively.

        Returns:
            torch.Tensor: Output tensor after applying cross-attention.
        """
        # x is of shape (batch_size, channels, depth/height, height, width) or (batch_size, channels, height, width)
        batch_size, channels, *spatial_dims = x.shape

        # Flatten spatial dimensions
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Calculate & spasify embedding
        pe = self.embedding(x)
        if self.gating is not None:
            pe = self.gating(pe)
        if self.multi_enc:
            pe = pe.sum(1)
        if len(pe.shape) > 3:
            pe = pe.squeeze(1)

        # Apply cross-attention using query=x, key=pe, value=pe
        attn_output, _ = self.attention(x, pe, pe)
        return attn_output.permute(0, 2, 1).view(batch_size, channels, *spatial_dims)
