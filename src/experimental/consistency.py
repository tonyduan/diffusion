from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from src.blocks import unsqueeze_as


@dataclass(frozen=True)
class ConsistencyModelConfig:

    train_steps_limit: int
    loss_type: str = "l2"
    s_0: int = 10
    s_1: int = 1280
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    p_mean: float = -1.1
    p_std: float = 2.0

    def __post_init__(self):
        assert self.s_0 <= self.s_1
        assert self.sigma_min <= self.sigma_max
        assert self.loss_type in ("l1", "l2")


class ConsistencyModel(nn.Module):

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_module: nn.Module,
        config: ConsistencyModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module

        # Input shape must be either (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4)

        # Unpack config and pre-compute a few relevant constants
        self.p_mean = config.p_mean
        self.p_std = config.p_std
        self.s_0 = config.s_0
        self.s_1 = config.s_1
        self.sigma_data = config.sigma_data
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.rho = config.rho
        self.loss_type = config.loss_type
        self.train_steps_limit = config.train_steps_limit
        self.sigma_min_root = (self.sigma_min) ** (1 / self.rho)
        self.sigma_max_root = (self.sigma_max) ** (1 / self.rho)
        self.k_prime = math.floor(self.train_steps_limit / (math.log2(self.s_1 / self.s_0) + 1))
        self.pseudo_huber_c = 0.00054 * math.prod(input_shape) ** 0.5

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

        # First compute the amount of discretization (number of sigmas needed)
        train_step_number = min(train_step_number, self.train_steps_limit)
        num_sigmas = min(self.s_0 * 2 ** math.floor(train_step_number / self.k_prime), self.s_1) + 1

        # Discretize the sigma space
        linspace = torch.linspace(0, 1, num_sigmas, device=x.device)
        sigmas = (self.sigma_min_root + linspace * (self.sigma_max_root - self.sigma_min_root)) ** self.rho

        # Draw a sample of sigma wiht importance sampling
        sigmas_weights = (
            torch.erf((torch.log(sigmas[1:]) - self.p_mean) / (2 ** 0.5 * self.p_std)) -
            torch.erf((torch.log(sigmas[:-1]) - self.p_mean) / (2 ** 0.5 * self.p_std))
        )
        sampled_idxs = torch.multinomial(sigmas_weights, num_samples=bsz, replacement=True)
        sigma_t = sigmas[sampled_idxs]
        sigma_t_plus_1 = sigmas[sampled_idxs + 1]

        # Forward the student and teacher, ensuring dropout is identical for both
        eps = torch.randn_like(x)
        with torch.random.fork_rng():
            x_t_plus_1 = x + unsqueeze_as(sigma_t_plus_1, x) * eps
            pred_t_plus_1 = self.nn_module_wrapper(x_t_plus_1, sigma_t_plus_1)
        with torch.no_grad(), torch.random.fork_rng():
            x_t = x + unsqueeze_as(sigma_t, x) * eps
            pred_t = self.nn_module_wrapper(x_t, sigma_t)

        # Compute loss and corresponding weights
        if self.loss_type == "l2":
            loss = (0.5 * (pred_t_plus_1 - pred_t) ** 2)
        elif self.loss_type == "l1":
            loss = (pred_t_plus_1 - pred_t).abs()
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        loss_weights = 1 / (sigma_t_plus_1 - sigma_t)
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
        """
        assert num_sampling_timesteps >= 1

        linspace = torch.linspace(1, 0, num_sampling_timesteps + 1, device=device)
        sigmas = (self.sigma_min_root + linspace * (self.sigma_max_root - self.sigma_min_root)) ** self.rho

        sigma_start = torch.empty((bsz,), dtype=torch.int64, device=device)
        sigma_end = torch.empty((bsz,), dtype=torch.int64, device=device)

        x = torch.randn((bsz, *self.input_shape), device=device) * self.sigma_max_root
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *self.input_shape), device=device)
        samples[-1] = x * self.sigma_data / (self.sigma_max ** 2 + self.sigma_data ** 2) ** 0.5

        for idx, (scalar_sigma_start, scalar_sigma_end) in enumerate(zip(sigmas[:-1], sigmas[1:])):

            sigma_start.fill_(scalar_sigma_start)
            sigma_end.fill_(scalar_sigma_end)

            x = self.nn_module_wrapper(x, sigma_start)
            eps = torch.randn_like(x)
            x += unsqueeze_as((sigma_end ** 2 - self.sigma_min ** 2).clamp(min=0) ** 0.5, x) * eps

            normalization_factor = self.sigma_data / (scalar_sigma_end ** 2 + self.sigma_data ** 2) ** 0.5
            samples[-1 - idx - 1] = x * normalization_factor

        return samples

