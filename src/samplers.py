from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from functools import cached_property

import torch


@dataclass
class InstantaneousPrediction:

    sigma: float
    x_t: torch.Tensor
    pred_x_0: torch.Tensor

    @cached_property
    def pred_eps(self):
        return (self.x_t - self.pred_x_0) / self.sigma


class Sampler(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, sigma_t: float, sigma_t_plus_1: float, pred_at_x_t: InstantaneousPrediction):
        """
        Take a step from sigma_t to sigma_t_plus_1.
        """
        pass


class EulerSampler(Sampler):

    def reset(self):
        pass

    def step(self, sigma_t: float, sigma_t_plus_1: float, pred_at_x_t: InstantaneousPrediction):
        return pred_at_x_t.x_t + pred_at_x_t.pred_eps * (sigma_t_plus_1 - sigma_t)


class MultistepDPMSampler(Sampler):

    def reset(self):
        self.history: deque[InstantaneousPrediction] = deque(maxlen=2)  # Order k=3

    def step(self, sigma_t: float, sigma_t_plus_1: float, pred_at_x_t: InstantaneousPrediction):
        if len(self.history) == 0:
            x_t_plus_1 = self.step_first_order(sigma_t, sigma_t_plus_1, pred_at_x_t)
        elif len(self.history) == 1:
            x_t_plus_1 = self.step_second_order(sigma_t, sigma_t_plus_1, pred_at_x_t)
        else:
            x_t_plus_1 = self.step_third_order(sigma_t, sigma_t_plus_1, pred_at_x_t)
        self.history.append(pred_at_x_t)
        return x_t_plus_1

    def step_first_order(self, sigma_t: float, sigma_t_plus_1: float, pred_at_x_t: InstantaneousPrediction):
        d_sigma = sigma_t_plus_1 - sigma_t
        return pred_at_x_t.x_t + pred_at_x_t.pred_eps * d_sigma

    def step_second_order(self, sigma_t: float, sigma_t_plus_1: float, pred_at_x_t: InstantaneousPrediction):
        pred_at_x_t_minus_1 = self.history[-1]
        d_sigma = sigma_t_plus_1 - sigma_t
        pred_first_derivative = (
            (pred_at_x_t.pred_eps - pred_at_x_t_minus_1.pred_eps) /
            (pred_at_x_t.sigma - pred_at_x_t_minus_1.sigma)
        )
        return (
            pred_at_x_t.x_t +
            pred_at_x_t.pred_eps * d_sigma +
            (1/2) * pred_first_derivative * d_sigma ** 2
        )

    def step_third_order(self, sigma_t: float, sigma_t_plus_1: float, pred_at_x_t: InstantaneousPrediction):
        pred_at_x_t_minus_1 = self.history[-1]
        pred_at_x_t_minus_2 = self.history[-2]
        d_sigma = sigma_t_plus_1 - sigma_t
        pred_first_derivative = (
            (pred_at_x_t.pred_eps - pred_at_x_t_minus_1.pred_eps) /
            (pred_at_x_t.sigma - pred_at_x_t_minus_1.sigma)
        )
        pred_first_derivative_past = (
            (pred_at_x_t_minus_1.pred_eps - pred_at_x_t_minus_2.pred_eps) /
            (pred_at_x_t_minus_1.sigma - pred_at_x_t_minus_2.sigma)
        )
        pred_second_derivative = (
            (pred_first_derivative - pred_first_derivative_past) /
            (0.5 * (pred_at_x_t.sigma + pred_at_x_t_minus_1.sigma) -
             0.5 * (pred_at_x_t_minus_1.sigma + pred_at_x_t_minus_2.sigma))
        )
        return (
            pred_at_x_t.x_t +
            pred_at_x_t.pred_eps * d_sigma +
            (1/2) * pred_first_derivative * d_sigma ** 2 +
            (1/6) * pred_second_derivative * d_sigma ** 3
        )
