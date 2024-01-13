from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.blocks import unsqueeze_as
from src.samplers import InstantaneousPrediction, Sampler, EulerSampler
from src.schedules import CosineSchedule, NoiseSchedule


@dataclass(frozen=True)
class ScoreMatchingModelConfig:

    # Network configuration and loss weighting
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5

    # Training time configuration
    loss_type: str = "l2"
    loss_weighting_type: str = "karras"
    train_sigma_schedule: NoiseSchedule = CosineSchedule()

    # Inference time configuration
    sampler: Sampler = EulerSampler()
    test_sigma_schedule: NoiseSchedule = CosineSchedule()

    def __post_init__(self):
        assert 0 <= self.sigma_min <= self.sigma_max
        assert self.sigma_min == self.train_sigma_schedule.sigma_min
        assert self.sigma_min == self.test_sigma_schedule.sigma_min
        assert self.sigma_max == self.train_sigma_schedule.sigma_max
        assert self.sigma_max == self.test_sigma_schedule.sigma_max
        assert self.loss_type in ("l1", "l2")
        assert self.loss_weighting_type in ("ones", "snr", "karras", "min_snr")


class ScoreMatchingModel(nn.Module):

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_module: nn.Module,
        config: ScoreMatchingModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module

        # Input shape must be either (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4)

        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.loss_type = config.loss_type
        self.loss_weighting_type = config.loss_weighting_type
        self.train_sigma_schedule = config.train_sigma_schedule
        self.test_sigma_schedule = config.test_sigma_schedule
        self.sampler = config.sampler

    def nn_module_wrapper(self, x, sigma, num_discrete_chunks=10000):
        """
        This function does two things:
        1. Implements Karras et al. 2022 pre-conditioning.
        2. Converts sigma in range [sigma_min, sigma_max] into a discrete input.

        Parameters
        ----------
          x: (bsz, *self.input_shape)
          sigma: (bsz,)
        """
        c_skip = self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2)
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_out = sigma * self.sigma_data / (self.sigma_data ** 2 + sigma ** 2) ** 0.5
        c_skip, c_in, c_out = unsqueeze_as(c_skip, x), unsqueeze_as(c_in, x), unsqueeze_as(c_out, x)
        log_sigmas_percentile = (
            (torch.log(sigma) - math.log(self.sigma_min)) /
            (math.log(self.sigma_max) - math.log(self.sigma_min))
        )
        sigmas_discrete = torch.floor(num_discrete_chunks * log_sigmas_percentile)
        sigmas_discrete = sigmas_discrete.clamp_(min=0, max=num_discrete_chunks - 1).long()
        return c_out * self.nn_module(c_in * x, sigmas_discrete) + c_skip * x

    def loss(self, x):
        """
        Returns
        -------
        loss: (bsz, *input_shape)
        """
        bsz, *_ = x.shape

        rng = torch.rand((bsz,), device=x.device)
        sigma = self.train_sigma_schedule.get_sigma_ppf(rng)

        x_t = x + unsqueeze_as(sigma, x) * torch.randn_like(x)
        pred = self.nn_module_wrapper(x_t, sigma)

        if self.loss_type == "l2":
            loss = (0.5 * (x - pred) ** 2)
        elif self.loss_type == "l1":
            loss = (x - pred).abs()
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        if self.loss_weighting_type == "ones":
            loss_weights = torch.ones_like(sigma)
        elif self.loss_weighting_type == "snr":
            loss_weights = self.sigma_data ** 2 / sigma ** 2
        elif self.loss_weighting_type == "min_snr":
            loss_weights = torch.clamp(self.sigma_data ** 2 / sigma ** 2, max=5.0)
        elif self.loss_weighting_type == "karras":
            loss_weights = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        else:
            raise AssertionError(f"Invalid {self.loss_weighting_type=}.")

        loss *= unsqueeze_as(loss_weights, loss)
        return loss

    @torch.no_grad()
    def sample(self, bsz, device, num_sampling_timesteps: int):
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
        This is deterministic for now, solving the probability flow ODE.
        """
        assert num_sampling_timesteps >= 1

        linspace = torch.linspace(1.0, 0.0, num_sampling_timesteps + 1, device=device)
        sigmas = self.test_sigma_schedule.get_sigma_ppf(linspace)

        sigma_start = torch.empty((bsz,), device=device)
        sigma_end = torch.empty((bsz,), device=device)

        x = torch.randn((bsz, *self.input_shape), device=device) * self.sigma_max
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *self.input_shape), device=device)
        samples[-1] = x * self.sigma_data / (self.sigma_max ** 2 + self.sigma_data ** 2) ** 0.5

        self.sampler.reset()

        for idx, (scalar_sigma_start, scalar_sigma_end) in enumerate(zip(sigmas[:-1], sigmas[1:])):

            sigma_start.fill_(scalar_sigma_start)
            sigma_end.fill_(scalar_sigma_end)

            pred_x_0 = self.nn_module_wrapper(x, sigma_start)
            pred_at_x_t = InstantaneousPrediction(scalar_sigma_start, x, pred_x_0)
            x = self.sampler.step(scalar_sigma_start, scalar_sigma_end, pred_at_x_t)

            normalization_factor = (
                self.sigma_data / (scalar_sigma_end ** 2 + self.sigma_data ** 2) ** 0.5
            )
            samples[-1 - idx - 1] = x * normalization_factor

        return samples
