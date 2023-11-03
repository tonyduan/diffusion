from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn

from src.blocks import unsqueeze_as


@dataclass(frozen=True)
class ScoreMatchingModelConfig:

    loss_type: str = "l2"
    sigma_schedule_type: str = "cosine"
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    p_mean: float | None = None
    p_std: float | None = None

    def __post_init__(self):
        assert self.sigma_min <= self.sigma_max
        assert self.loss_type in ("l1", "l2")
        assert self.sigma_schedule_type in ("cosine", "lognormal")
        assert (self.sigma_schedule_type == "lognormal") ^ (self.p_mean is None and self.p_std is None)


class ScoreMatchingModel(nn.Module):

    def __init__(
        self,
        input_shape: Tuple,
        nn_module: nn.Module,
        config: ScoreMatchingModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module

        # Input shape must be either (c,) or (c, h, w)
        assert len(input_shape) in (1, 3)

        # Unpack config and pre-compute a few relevant constants
        self.p_mean = config.p_mean
        self.p_std = config.p_std
        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.rho = config.rho
        self.loss_type = config.loss_type
        self.sigma_schedule_type = config.sigma_schedule_type
        self.sigma_min_root = (self.sigma_min) ** (1 / self.rho)
        self.sigma_max_root = (self.sigma_max) ** (1 / self.rho)

    def nn_module_wrapper(self, x, sigma, num_discrete_chunks=10000):
        """
        This function does two things:
        1. Implements Karras et al. 2022 pre-conditioning
        2. Converts sigma which have range (sigma_min, sigma_max) into a discrete input

        Parameters
        ----------
          x: (bsz, *self.input_shape)
          sigma: (bsz,)
        """
        c_skip = self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_out = sigma * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2) ** 0.5
        c_skip, c_in, c_out = unsqueeze_as(c_skip, x), unsqueeze_as(c_in, x), unsqueeze_as(c_out, x)
        sigmas_percentile = (
            ((sigma ** (1 / self.rho)) - self.sigma_min_root) / (self.sigma_max_root - self.sigma_min_root)
        )
        sigmas_discrete = torch.floor(num_discrete_chunks * sigmas_percentile).clamp(max=num_discrete_chunks - 1).long()
        return c_out * self.nn_module(c_in * x, sigmas_discrete) + c_skip * x

    def loss(self, x, train_step_number: int):
        """
        Returns
        -------
        loss: (bsz, *input_shape)
        """
        bsz, *_ = x.shape

        if self.sigma_schedule_type == "cosine":
            logsnr_min = -2 * (math.log(self.sigma_min) - math.log(self.sigma_data))
            logsnr_max = -2 * (math.log(self.sigma_max) - math.log(self.sigma_data))
            t_min = math.atan(math.exp(-0.5 * logsnr_max))
            t_max = math.atan(math.exp(-0.5 * logsnr_min))
            sigma = -2 * torch.log(torch.tan(t_min + torch.rand((bsz,), device=x.device) * (t_max - t_min)))
            sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        elif self.sigma_schedule_type == "lognormal":
            sigma = torch.exp(self.p_std * torch.randn((bsz,), device=x.device) + self.p_mean)
        else:
            raise AssertionError(f"Invalid {self.sigma_schedule_type=}.")

        x_t = x + unsqueeze_as(sigma, x) * torch.randn_like(x)
        pred = self.nn_module_wrapper(x_t, sigma)

        if self.loss_type == "l2":
            loss = (0.5 * (x - pred) ** 2)
        elif self.loss_type == "l1":
            loss = (x - pred).abs()
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        loss_weights = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss *= unsqueeze_as(loss_weights, loss)
        return loss

    @torch.no_grad()
    def sample(self, bsz, device, num_sampling_timesteps: int, use_heun_step: bool = False):
        """
        Parameters
        ----------
        num_sampling_timesteps: number of steps to take between sigma_max to sigma_min

        Returns
        -------
        samples: (sampling_timesteps + 1, bsz, *self.input_shape)
            index 0 corresponds to x_0
            index t corresponds to x_t
            last index corresponds to random noise

        Notes
        -----
        This is deterministic for now, need to add stochastic sampler.
        """
        assert num_sampling_timesteps >= 1

        linspace = torch.linspace(1, 0, num_sampling_timesteps + 1, device=device)
        sigmas = (self.sigma_min_root + linspace * (self.sigma_max_root - self.sigma_min_root)) ** self.rho

        sigma_start = torch.empty((bsz,), dtype=torch.int64, device=device)
        sigma_end = torch.empty((bsz,), dtype=torch.int64, device=device)

        x = torch.randn((bsz, *self.input_shape), device=device) * self.sigma_max
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *self.input_shape), device=device)
        samples[-1] = x * self.sigma_data / (self.sigma_max ** 2 + self.sigma_data ** 2) ** 0.5

        for idx, (scalar_sigma_start, scalar_sigma_end) in enumerate(zip(sigmas[:-1], sigmas[1:])):

            sigma_start.fill_(scalar_sigma_start)
            sigma_end.fill_(scalar_sigma_end)

            pred_x_0 = self.nn_module_wrapper(x, sigma_start)
            dx_dsigma = (x - pred_x_0) / scalar_sigma_start
            dsigma = scalar_sigma_end - scalar_sigma_start

            if use_heun_step and scalar_sigma_end > 0:
                x_ = x + dx_dsigma * dsigma
                pred_x_0_ = self.nn_module_wrapper(x_, sigma_end)
                dx_dsigma_ = (x_ - pred_x_0_) / scalar_sigma_end
                x = x + 0.5 * (dx_dsigma + dx_dsigma_) * dsigma
            else:
                x = x + dx_dsigma * dsigma

            normalization_factor = self.sigma_data / (scalar_sigma_end ** 2 + self.sigma_data ** 2) ** 0.5
            samples[-1 - idx - 1] = x * normalization_factor

        return samples

