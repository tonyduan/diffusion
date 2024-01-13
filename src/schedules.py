from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
import math

import torch


@dataclass(frozen=True)
class NoiseSchedule(ABC):

    @abstractmethod
    def get_sigma_ppf(self, p: torch.Tensor, sigma_min: float, sigma_max: float):
        """
        Transform p in [0, 1] to sigma between [sigma_min, sigma_max] via icdf.
        """
        pass


@dataclass(frozen=True)
class CosineSchedule(NoiseSchedule):

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    kappa: float = 1.0

    @cached_property
    def theta_min(self):
        logsnr_min = 2 * (math.log(self.kappa) - math.log(self.sigma_min))
        return math.atan(math.exp(-0.5 * logsnr_min))

    @cached_property
    def theta_max(self):
        logsnr_max = 2 * (math.log(self.kappa) - math.log(self.sigma_max))
        return math.atan(math.exp(-0.5 * logsnr_max))

    def get_sigma_ppf(self, p: torch.Tensor):
        return torch.tan(self.theta_min + p * (self.theta_max - self.theta_min)) / self.kappa


@dataclass(frozen=True)
class ExponentialSchedule(NoiseSchedule):

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0

    @cached_property
    def inv_sigma_min(self):
        return self.sigma_min ** (1 / self.rho)

    @cached_property
    def inv_sigma_max(self):
        return self.sigma_max ** (1 / self.rho)

    def get_sigma_ppf(self, p: torch.Tensor):
        return (self.inv_sigma_min + p * (self.inv_sigma_max - self.inv_sigma_min)) ** self.rho


@dataclass(frozen=True)
class NormalSchedule(NoiseSchedule):

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    mean: float = -1.2
    std: float = 1.2

    @cached_property
    def erf_sigma_min(self):
        return math.erf(math.log(self.sigma_min))

    @cached_property
    def erf_sigma_max(self):
        return math.erf(math.log(self.sigma_max))

    def get_sigma_ppf(self, p: torch.Tensor):
        z = torch.special.erfinv(self.erf_sigma_min + p * (self.erf_sigma_max - self.erf_sigma_min))
        return torch.exp(z * self.std + self.mean)

