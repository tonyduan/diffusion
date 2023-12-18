from abc import ABC, abstractmethod


class Sampler(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, x_t, pred_x_0, sigma_t, sigma_t_plus_1):
        """
        Take a step from sigma_t to sigma_t_plus_1.
        """
        pass


class EulerSampler(Sampler):
    #
    # Integrate in sigma space. Straightforward discretization.
    #
    def reset(self):
        pass

    def step(self, x_t, pred_x_0, sigma_t, sigma_t_plus_1):
        pred_eps = (x_t - pred_x_0) / sigma_t
        x_t_plus_1 = x_t + pred_eps * (sigma_t_plus_1 - sigma_t)
        return x_t_plus_1


class MultistepDPMSampler(Sampler):
    #
    # Integrate in log(sigma) space. Taylor approximation exponential integrator.
    #
    def reset(self):
        self.history = []

    def step(self, x_t, pred_x_0, sigma_t, sigma_t_plus_1):
        pass

