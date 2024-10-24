import numpy as np
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

from windowing import (
    window_partition_2d, 
    window_partition_3d, 
    window_reverse_2d, 
    window_reverse_3d
)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=2048):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dim_head, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x)
        if mask is not None:
            x = x * mask  # Apply mask before attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class SWinTr(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            dropout,
            data_type='2D'
        ):
        super(SWinTr, self).__init__()
        self.data_type = data_type
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        if self.data_type == '1D':
            D = x.shape[-1]
            N, C = x.shape[:2]
        elif self.data_type == '2D':
            H, W = x.shape[-2:]
            N, C = x.shape[:2]
        elif self.data_type == '3D':
            H, W, D = x.shape[-3:]
            N, C = x.shape[:2]

        # extract dimensionality
        dim = int(self.data_type[0])

        # Shift windows
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=dim*[-self.shift_size],
                dims=(np.arange(dim)+2).astype(int).tolist()
            )

        # Window partition
        if self.data_type == '2D':
            x_windows = window_partition_2d(x, self.window_size).permute(0, 2, 3, 1)
        elif self.data_type == '3D':
            x_windows = window_partition_3d(x, self.window_size).permute(0, 2, 3, 4, 1)

        x_windows = x_windows.view(
            N, -1,
            int(np.prod(dim*[self.window_size]))
        )
        attn_windows, _ = self.attn(x_windows, x_windows, x_windows)[0]
        attn_windows = attn_windows.view(N, -1, dim*[self.window_size])

        if self.data_type == '2D':
            x = window_reverse_2d(attn_windows, self.window_size, H, W)
        elif self.data_type == '3D':
            x = window_reverse_3d(attn_windows, self.window_size, H, W, D)

        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=[self.shift_size]*dim,
                dims=(np.arange(dim)+2).astype(int).tolist()
            )

        x = x.view(N, -1)
        x = x + self.norm2(x)
        x = x + self.mlp(x)
        return x


