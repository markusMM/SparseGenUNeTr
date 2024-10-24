import torch
import torch.nn as nn


class BernoulliGating(nn.Module):
    def __init__(self, num_patches: int, num_channels: int, num_particles: int, use_bayesian: bool = True):
        """
        Bayesian gating method for selecting and recombining multiple encodings.

        Args:
            num_patches: Number of patches.
            num_channels: Number of channels per patch.
            num_encodings: Number of encodings.
            use_bayesian: Boolean to enable Bayesian sparsification.
        """
        super(BernoulliGating, self).__init__()
        self.num_particles = num_particles
        self.use_bayesian = use_bayesian

        if use_bayesian:
            # Learnable z_filter for Bayesian sparsification
            self.z_filter = nn.Parameter(torch.ones(num_particles, num_patches, num_channels))
            self.state = torch.ones_like(self.z_filter)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        # Apply Bayesian gating
        if self.use_bayesian:
            self.state = torch.bernoulli(torch.sigmoid(self.z_filter))  # Sample latent z ~ Bernoulli(p)
            encodings = encodings * self.state

        # Normalize and sum encodings
        normalized_sum = torch.sum(encodings, dim=1) / self.num_particles
        return normalized_sum
