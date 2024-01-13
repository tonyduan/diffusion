"""
This file provides a simple, self-contained implementation of DDIM (with DDPM as a special case).
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.blocks import unsqueeze_to


@dataclass(frozen=True)
class DiffusionModelConfig:

    num_timesteps: int
    target_type: str = "pred_eps"
    noise_schedule_type: str = "cosine"
    loss_type: str = "l2"
    gamma_type: float = "ddim"

    def __post_init__(self):
        assert self.num_timesteps > 0
        assert self.target_type in ("pred_x_0", "pred_eps", "pred_v")
        assert self.noise_schedule_type in ("linear", "cosine")
        assert self.loss_type in ("l1", "l2")
        assert self.gamma_type in ("ddim", "ddpm")


class DiffusionModel(nn.Module):

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_module: nn.Module,
        config: DiffusionModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module
        self.num_timesteps = config.num_timesteps
        self.target_type = config.target_type
        self.gamma_type = config.gamma_type
        self.noise_schedule_type = config.noise_schedule_type
        self.loss_type = config.loss_type

        # Input shape must be either (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4)

        # Construct the noise schedule
        if self.noise_schedule_type == "linear":
            beta_t = torch.linspace(1e-4, 2e-2, self.num_timesteps + 1)
            alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5
        elif self.noise_schedule_type == "cosine":
            linspace = torch.linspace(0, 1, self.num_timesteps + 1)
            f_t = torch.cos((linspace + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            bar_alpha_t = f_t / f_t[0]
            beta_t = torch.zeros_like(bar_alpha_t)
            beta_t[1:] = (1 - (bar_alpha_t[1:] / bar_alpha_t[:-1])).clamp(min=0, max=0.999)
            alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5
        else:
            raise AssertionError(f"Invalid {self.noise_schedule_type=}.")

        # These tensors are shape (num_timesteps + 1, *self.input_shape)
        # For example, 2D: (num_timesteps + 1, 1, 1, 1)
        #              1D: (num_timesteps + 1, 1)
        alpha_t = unsqueeze_to(alpha_t, len(self.input_shape) + 1)
        sigma_t = (1 - alpha_t ** 2).clamp(min=0) ** 0.5
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("sigma_t", sigma_t)

    def loss(self, x: torch.Tensor):
        """
        Returns
        -------
        loss: (bsz, *input_shape)
        """
        bsz, *_ = x.shape
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)
        eps = torch.randn_like(x)
        x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps
        pred_target = self.nn_module(x_t, t_sample)

        if self.target_type == "pred_x_0":
            gt_target = x
        elif self.target_type == "pred_eps":
            gt_target = eps
        elif self.target_type == "pred_v":
            gt_target = self.alpha_t[t_sample] * eps - self.sigma_t[t_sample] * x
        else:
            raise AssertionError(f"Invalid {self.target_type=}.")

        if self.loss_type == "l2":
            loss = 0.5 * (gt_target - pred_target) ** 2
        elif self.loss_type == "l1":
            loss = torch.abs(gt_target - pred_target)
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        return loss

    @torch.no_grad()
    def sample(self, bsz: int, device: str, num_sampling_timesteps: int | None = None):
        """
        Parameters
        ----------
        num_sampling_timesteps: int. If unspecified, defaults to self.num_timesteps.

        Returns
        -------
        samples: (num_sampling_timesteps + 1, bsz, *self.input_shape)
            index 0 corresponds to x_0
            index t corresponds to x_t
            last index corresponds to random noise
        """
        num_sampling_timesteps = num_sampling_timesteps or self.num_timesteps
        assert 1 <= num_sampling_timesteps <= self.num_timesteps

        x = torch.randn((bsz, *self.input_shape), device=device)
        t_start = torch.empty((bsz,), dtype=torch.int64, device=device)
        t_end = torch.empty((bsz,), dtype=torch.int64, device=device)

        subseq = torch.linspace(self.num_timesteps, 0, num_sampling_timesteps + 1).round()
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *self.input_shape), device=device)
        samples[-1] = x

        # Note that t_start > t_end we're traversing pairwise down subseq.
        # For example, subseq here could be [500, 400, 300, 200, 100, 0]
        for idx, (scalar_t_start, scalar_t_end) in enumerate(zip(subseq[:-1], subseq[1:])):

            t_start.fill_(scalar_t_start)
            t_end.fill_(scalar_t_end)
            noise = torch.zeros_like(x) if scalar_t_end == 0 else torch.randn_like(x)

            if self.gamma_type == "ddim":
                gamma_t = 0.0
            elif self.gamma_type == "ddpm":
                gamma_t = (
                    self.sigma_t[t_end] / self.sigma_t[t_start] *
                    (1 - self.alpha_t[t_start] ** 2 / self.alpha_t[t_end] ** 2) ** 0.5
                )
            else:
                raise AssertionError(f"Invalid {self.gamma_type=}.")

            nn_out = self.nn_module(x, t_start)
            if self.target_type == "pred_x_0":
                pred_x_0 = nn_out
                pred_eps = (x - self.alpha_t[t_start] * nn_out) / self.sigma_t[t_start]
            elif self.target_type == "pred_eps":
                pred_x_0 = (x - self.sigma_t[t_start] * nn_out) / self.alpha_t[t_start]
                pred_eps = nn_out
            elif self.target_type == "pred_v":
                pred_x_0 = self.alpha_t[t_start] * x - self.sigma_t[t_start] * nn_out
                pred_eps = self.sigma_t[t_start] * x + self.alpha_t[t_start] * nn_out
            else:
                raise AssertionError(f"Invalid {self.target_type=}.")

            x = (
                (self.alpha_t[t_end] * pred_x_0) +
                (self.sigma_t[t_end] ** 2 - gamma_t ** 2).clamp(min=0) ** 0.5 * pred_eps +
                (gamma_t * noise)
            )
            samples[-1 - idx - 1] = x

        return samples

