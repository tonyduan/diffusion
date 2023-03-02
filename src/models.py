from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn

from src.blocks import unsqueeze_to


@dataclass(frozen=True)
class DiffusionModelConfig:

    target_type: str = "pred_eps"
    sigma_type: str = "upper_bound"
    noise_schedule_type: str = "linear"

    def __post_init__(self):
        assert self.target_type in ("pred_x_0", "pred_eps")
        assert self.sigma_type in ("lower_bound", "upper_bound")
        assert self.noise_schedule_type in ("linear", "cosine")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        num_timesteps: int,
        input_shape: Tuple,
        nn_module: nn.Module,
        config: DiffusionModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_timesteps = num_timesteps
        self.nn_module = nn_module
        self.target_type = config.target_type
        self.sigma_type = config.sigma_type
        self.noise_schedule_type = config.noise_schedule_type

        # Input shape must be either (c,) or (c, h, w)
        assert len(input_shape) in (1, 3)

        if self.noise_schedule_type == "linear":
            beta = torch.linspace(1e-4, 2e-2, num_timesteps + 1)
        if self.noise_schedule_type == "cosine":
            rng = torch.arange(num_timesteps + 1)
            f_t = torch.cos((rng / num_timesteps + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            bar_alpha = f_t / f_t[0]
            beta = torch.zeros_like(bar_alpha)
            beta[1:] = (1 - (bar_alpha[1:] / bar_alpha[:-1])).clip(0, 0.999)

        # These tensors are shape (num_timesteps + 1, 1, 1, 1) if 2D or (num_timesteps + 1, 1) if 1D
        beta = unsqueeze_to(beta, len(self.input_shape) + 1)
        self.register_buffer("bar_alpha", torch.cumprod(1 - beta, dim=0))

    def loss(self, x):
        """
        Returns
        -------
        loss: (bsz,)
        """
        bsz = len(x)
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)
        eps = torch.randn_like(x)
        x_t = (
            torch.sqrt(self.bar_alpha[t_sample]) * x +
            torch.sqrt(1 - self.bar_alpha[t_sample]) * eps
        )
        pred_target = self.nn_module(x_t, t_sample)
        if self.target_type == "pred_x_0":
            loss = torch.mean((x - pred_target) ** 2, dim=1)
        if self.target_type == "pred_eps":
            loss = torch.mean((eps - pred_target) ** 2, dim=1)
        return loss

    @torch.no_grad()
    def sample(self, bsz, device, sampling_timesteps=None):
        """
        Parameters
        ----------
        sampling_timesteps: if unspecified, defaults to self.num_timesteps

        Returns
        -------
        samples: (sampling_timesteps + 1, bsz, *input_shape)
            index 0 corresponds to x_0
            index t corresponds to x_t
            last index corresponds to random noise
        """
        sampling_timesteps = self.num_timesteps if sampling_timesteps is None else sampling_timesteps
        assert 1 <= sampling_timesteps <= self.num_timesteps

        x = torch.randn((bsz, *self.input_shape), device=device)
        t_start = torch.empty((bsz,), dtype=torch.int64, device=device)
        t_end = torch.empty((bsz,), dtype=torch.int64, device=device)

        subseq = torch.linspace(self.num_timesteps, 0, sampling_timesteps + 1).round()
        samples = torch.zeros((sampling_timesteps + 1, bsz, *self.input_shape), device=device)
        samples[-1] = x

        for idx, (scalar_t_start, scalar_t_end) in enumerate(zip(subseq[:-1], subseq[1:])):

            t_start.fill_(scalar_t_start)
            t_end.fill_(scalar_t_end)
            noise = torch.randn_like(x) if scalar_t_end > 0 else torch.zeros_like(x)

            if self.target_type == "pred_x_0":
                pred_x_0 = self.nn_module(x, t_start)
            if self.target_type == "pred_eps":
                pred_eps = self.nn_module(x, t_start)
                pred_x_0 = (
                    x - torch.sqrt(1 - self.bar_alpha[t_start]) * pred_eps
                ) / torch.sqrt(self.bar_alpha[t_start])

            x = (
                torch.sqrt(self.bar_alpha[t_end]) * (1 - self.bar_alpha[t_start] / self.bar_alpha[t_end]) * pred_x_0 +
                (1 - self.bar_alpha[t_end]) * torch.sqrt(self.bar_alpha[t_start] / self.bar_alpha[t_end]) * x
            ) / (1 - self.bar_alpha[t_start])
            if self.sigma_type == "upper_bound":
                x += torch.sqrt(1 - self.bar_alpha[t_start] / self.bar_alpha[t_end]) * noise
            if self.sigma_type == "lower_bound":
                x += torch.sqrt(
                    (1 - self.bar_alpha[t_start] / self.bar_alpha[t_end]) *
                    (1 - self.bar_alpha[t_end]) / (1 - self.bar_alpha[t_start])
                ) * noise

            samples[-1 - idx - 1] = x
        return samples

