import math
import numpy as np
from matplotlib import pyplot as plt


def get_linear_schedule(num_timesteps):
    beta_t = np.linspace(1e-4, 2e-2, num_timesteps + 1)
    alpha_t = np.cumprod(1 - beta_t, axis=0) ** 0.5
    return alpha_t

def get_cosine_schedule(num_timesteps):
    linspace = np.linspace(0, 1, num_timesteps + 1)
    f_t = np.cos((linspace + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
    bar_alpha_t = f_t / f_t[0]
    beta_t = np.zeros_like(bar_alpha_t)
    beta_t[1:] = np.clip(1 - (bar_alpha_t[1:] / bar_alpha_t[:-1]), a_min=0, a_max=0.999)
    alpha_t = np.cumprod(1 - beta_t, axis=0) ** 0.5
    return alpha_t


if __name__ == "__main__":

    T = 500
    T_rng = np.arange(T + 1)

    alpha_linear = get_linear_schedule(T)
    alpha_cosine = get_cosine_schedule(T)

    sigma_linear = (1 - alpha_linear ** 2) ** 0.5
    sigma_cosine = (1 - alpha_cosine ** 2) ** 0.5

    logsnr_linear = 2 * (np.log(alpha_linear) - np.log(sigma_linear))
    logsnr_cosine = 2 * (np.log(alpha_cosine) - np.log(sigma_cosine))

    plt.figure(figsize=(7, 2.6), dpi=200)

    plt.subplot(1, 3, 1)
    plt.plot(T_rng, alpha_linear, label="Linear", color="#00204E")
    plt.plot(T_rng, alpha_cosine, label="Cosine", color="#800000")
    plt.ylabel("$\\alpha_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0,1])
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 2)
    plt.plot(T_rng, sigma_linear, label="Linear", color="#00204E")
    plt.plot(T_rng, sigma_cosine, label="Cosine", color="#800000")
    plt.ylabel("$\\sigma_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0, 1])
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 3)
    plt.plot(T_rng, logsnr_linear, label="Linear", color="#00204E")
    plt.plot(T_rng, logsnr_cosine, label="Cosine", color="#800000")
    plt.ylabel("$\\log SNR_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([-10, 10])
    plt.ylim([-11, 11])
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("./figures/fig_schedules.png")

