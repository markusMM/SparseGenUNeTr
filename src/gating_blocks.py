from typing import Iterable, Union
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class BernoulliGating(nn.Module):
    def __init__(
        self, 
        num_patches: int, 
        num_channels: int, 
        num_particles: int, 
        use_bayesian: bool = True,
        p_default: Union[torch.Tensor, float] = .3,
        patchvise: bool = False
    ):
        """
        Bayesian gating method for selecting and recombining multiple data channels.

        Args:
            num_patches: Number of patches.
            num_channels: Number of channels per patch.
            num_encodings: Number of encodings.
            num_particles: Number of samples on each filtering.
            use_bayesian: Boolean to enable Bayesian sparsification.
        """
        super(BernoulliGating, self).__init__()
        self.num_particles = num_particles
        self.use_bayesian = use_bayesian
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.patchvise = patchvise

        p_shape = (num_patches, num_channels) if patchvise else (num_channels,)

        if not isinstance(p_default, torch.Tensor):
            p_default = torch.tensor(p_default)
        p_default = torch.log(p_default / (1 - p_default))  # logit for NN to learn

        if len(p_default.shape) > 0:
            if p_default.shape[0] != num_channels: # type: ignore
                warnings.warn(
                    f"Shape of p-spike \"{p_default.shape}\" \
                    does not aling with the model shape \
                    \"{(num_channels,)}\"!"
                )
                p_default = p_default[0]

        if p_default.numel() == 1:
            p_default = torch.ones(
                p_shape
            ) * p_default  # type: ignore

        if use_bayesian:
            # Learnable z_filter for Bayesian sparsification
            self.z_filter = nn.Parameter(p_default)
            self.state = torch.ones(num_particles, *p_shape)
        else:
            self.z_filter = p_default

    def pies(self):
        return torch.sigmoid(self.z_filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get data dimensions
        ni = (np.where(np.array(x.shape)==self.num_channels)[0]).item()
        ndim = len(x.shape) - ni - 1 # number of data axes in x

        # if we do lack of already drawn samples
        if x.shape[ni-2] != self.num_particles:
            x = x.unsqueeze(ni-2)

        # Apply Bayesian gating - z ~ Bernoulli(p) x (n_particles)
        z = Bernoulli(
            torch.sigmoid(self.z_filter)
        ).sample([self.num_particles] + [1]*ndim)  # type: ignore

        # if Baesian, store the states for later calculation
        if self.use_bayesian:
            self.state = z
        if not self.patchvise: 
            z = z.unsqueeze(-max(ndim, 1) - 1)  # type: ignore
        x = x * z  # The actual filter

        # Expectation values
        normalized_sum = torch.sum(x, dim=(ni - 2)) / self.num_particles
        return normalized_sum
