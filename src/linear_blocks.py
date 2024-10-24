import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from typing import Union


class SikeSlabLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 2048,
        pi_init: float = .36,
        va_init: float = .05,
        max_batch_size: int = 128
    ) -> None:
        super().__init__()
        if pi_init >= 1:
            pi_init = pi_init / output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.pi = nn.Parameter(torch.tensor(pi_init))
        self.va = nn.Parameter(va_init)
        self.w = nn.Parameter(
            torch.randn(input_dim, num_latents) / 4.0 + 
            torch.rand(input_dim, num_latents)
        )
        self.max_batch_size = max_batch_size
        self.k_max = 1000

    @staticmethod
    def k_sel_overflow(x: torch.Tensor, w_r: torch.Tensor, k: int) -> torch.Tensor:
        return torch.argsort((x @ w_r).abs())[-k:]

    def k_sel_underflow(self, x: torch.Tensor, w_r: torch.Tensor, k: int) -> torch.Tensor:
        k_max = max(k, min(self.k_max, len(w_r.T)))
        h_new = torch.randint(0, len(w_r.T), [k_max])
        return self.k_sel_overflow(x, w_r[:, h_new], k)
    
    def draw_prior(self) -> torch.Tensor:
        return Bernoulli(
            F.sigmoid(self.pi)
        ).sample((self.num_latents,))  # type: ignore

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.draw_prior()
        zp, _ = torch.where(z)
        zn, _ = torch.where(~z)
        if len(zp) > self.output_dim:
            k = len(zp) - self.output_dim
            z = torch.cat([zp, self.k_sel_overflow(x, self.w[:, zp], k)])
        elif len(zp) < self.output_dim:
            k = self.output_dim - len(zp)
            z = torch.cat([zp, self.k_sel_underflow(x, self.w[:, zn], k)])
        
        return x @ self.w[:, z], z
