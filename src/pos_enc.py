import torch
import torch.nn as nn
from typing import Union, Tuple

class MultipleEncoding(nn.Module):
    def __init__(self, num_encodings: int, embed_dim: int):
        """
        Base class for generating multiple types of positional encodings.

        Args:
            num_encodings (int): Number of different positional encodings to generate.
            embed_dim (int): Embedding dimension for each encoding.

        Attributes:
            num_encodings (int): The number of distinct positional encodings.
            embed_dim (int): The dimensionality of each embedding.
        """
        super(MultipleEncoding, self).__init__()
        self.num_encodings = num_encodings
        self.embed_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method to generate encodings.")


class MultiWeightedPositionalEncoding(MultipleEncoding):
    def __init__(self, num_patches: int, embed_dim: int, num_encodings: int = 3):
        """
        Free parameter positional encoding with multiple encoding variants. Each encoding
        is learned independently and can be used in transformer-like architectures for different
        forms of positional information.

        Args:
            num_patches (int): Number of patches or tokens in the input data (sequence length or spatial size).
            embed_dim (int): Dimensionality of the embedding for each token or patch.
            num_encodings (int, optional): Number of different sets of positional encodings. Default is 3.

        Attributes:
            position_embedding (nn.Parameter): Learnable parameters for positional encodings, 
                initialized randomly with shape `(num_encodings, 1, num_patches, embed_dim)`.
        """
        super(MultiWeightedPositionalEncoding, self).__init__(num_encodings, embed_dim)
        
        # Learnable parameters for multiple positional encodings
        self.position_embedding = nn.Parameter(torch.randn(num_encodings, 1, num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to generate multiple sets of learnable positional encodings.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, embed_dim, num_patches)`.

        Returns:
            torch.Tensor: Positional encodings of shape `(batch_size, num_encodings, num_patches, embed_dim)`.
        """
        batch_size = x.shape[0]
        
        # Repeat the learnable positional encodings for the batch size
        # (batch_size, num_encodings, num_patches, embed_dim)
        pos_embedding = self.position_embedding.repeat(batch_size, 1, 1, 1, 1)
        
        return pos_embedding


class MultiRotationalEncoding(MultipleEncoding):
    def __init__(self, kernel_size: Union[Tuple[int], int], embed_dim: int, num_encodings: int = 3, is_3d: bool = True):
        """
        Rotational positional encoding that supports generating multiple encoding sets.
        This method encodes spatial information using circular or spherical coordinates
        (radius, theta, phi) for 2D or 3D data.

        Args:
            kernel_size (Union[Tuple[int], int]): Size of the spatial dimensions (depth, height, width).
                If an integer is passed, it will be broadcast to the appropriate dimensions based on `is_3d`.
            embed_dim (int): Dimensionality of the embeddings.
            num_encodings (int, optional): Number of different sets of positional encodings. Default is 3.
            is_3d (bool, optional): Whether to use 3D positional encodings. If False, 2D encodings will be generated. Default is True.

        Attributes:
            theta (nn.Parameter): Learnable phase shift for the phi angle in spherical coordinates. 
                Shape depends on whether `is_3d` is True (3D encoding) or False (2D encoding).
            is_3d (bool): Flag indicating if the encoding is for 3D data (True) or 2D data (False).
        """
        super(MultiRotationalEncoding, self).__init__(num_encodings=num_encodings, embed_dim=embed_dim)
        n_dim = (3 if is_3d else 2)
        assert not (num_encodings % n_dim)
        num_encodings = num_encodings // n_dim
        self.is_3d = is_3d

        # Initialize theta as a learnable parameter for phase shifts
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * n_dim  # type: ignore
        if is_3d:
            depth, height, width = kernel_size  # type: ignore
            self.phi = nn.Parameter(torch.zeros(num_encodings, depth, height, width))
            self.theta = nn.Parameter(torch.zeros(num_encodings, depth, height, width))
        else:
            height, width = kernel_size  # type: ignore
            self.phi = nn.Parameter(torch.zeros(num_encodings, height, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to generate rotational positional encodings based on spherical or
        cylindrical coordinates. Encodes the spatial information as a combination of radius,
        polar angle (theta), and azimuthal angle (phi).

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, channels, depth, height, width)` for 3D or
                `(batch_size, channels, height, width)` for 2D.

        Returns:
            torch.Tensor: Positional encodings of shape `(batch_size, 3 x num_encodings, flattened_size)`.
                The flattened size depends on the input spatial dimensions.
        """
        batch_size, _, *spatial_dims = x.shape

        if self.is_3d:
            depth, height, width = spatial_dims
            zz, yy, xx = torch.meshgrid(torch.arange(depth), torch.arange(height), torch.arange(width))
            r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2 + zz.float() ** 2)
            theta = torch.atan2(torch.sqrt(xx.float() ** 2 + yy.float() ** 2), zz.float() + 1e-8) + self.theta
            phi = torch.atan2(yy.float(), xx.float()) + self.phi
            pe_r = (torch.sin(r) + torch.cos(r))[None]
            pe_theta = torch.sin(theta) + torch.cos(theta)
            pe_phi = torch.sin(phi) + torch.cos(phi)

            # Combine the encodings into the final representation
            encodings = torch.cat([pe_r, pe_theta, pe_phi], dim=0)  # Shape: (num_encodings x 3, ...)

        else:
            height, width = spatial_dims
            yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width))
            xx, yy = xx.to(x.device), yy.to(x.device)
            r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
            phi = torch.atan2(yy.float(), xx.float()) + self.phi
            
            pe_r = (torch.sin(r) + torch.cos(r))[None]
            pe_phi = torch.sin(phi) + torch.cos(phi)

            # Combine the encodings into the final representation
            encodings = torch.cat([pe_r, pe_phi], dim=0)  # Shape: (num_encodings x 3, ...)

        return encodings.view(self.num_encodings, -1).repeat(batch_size, 1, 1)
