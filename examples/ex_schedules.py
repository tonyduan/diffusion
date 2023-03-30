import math
import numpy as np
from matplotlib import pyplot as plt


def get_linear_schedule_beta(num_timesteps):
    return np.linspace(1e-4, 2e-2, num_timesteps + 1)

def get_cosine_schedule_beta(num_timesteps):
    rng = np.arange(num_timesteps + 1)
    f_t = np.cos((rng / num_timesteps + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
    bar_alpha = f_t / f_t[0]
    beta = np.zeros_like(bar_alpha)
    beta[1:] = (1 - (bar_alpha[1:] / bar_alpha[:-1])).clip(0, 0.999)
    return beta

if __name__ == "__main__":

    T = 500
    T_rng = np.arange(T + 1)

    beta_linear = get_linear_schedule_beta(T)
    beta_cosine = get_cosine_schedule_beta(T)

    bar_alpha_linear = np.cumprod(1 - beta_linear)
    bar_alpha_cosine = np.cumprod(1 - beta_cosine)

    snr_linear = bar_alpha_linear / (1 - bar_alpha_linear)
    snr_cosine = bar_alpha_cosine / (1 - bar_alpha_cosine)

    plt.figure(figsize=(7, 2.6), dpi=300)
    plt.subplot(1, 3, 1)
    plt.plot(T_rng, beta_linear, label="Linear", color="#00204E")
    plt.plot(T_rng, beta_cosine, label="Cosine", color="#800000")
    plt.ylabel("$\\beta_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0, 1])
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 2)
    plt.plot(T_rng, np.sqrt(bar_alpha_linear), label="Linear", color="#00204E")
    plt.plot(T_rng, np.sqrt(bar_alpha_cosine), label="Cosine", color="#800000")
    plt.ylabel("$\\sqrt{\\bar\\alpha_t}$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0,1])
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 3)
    plt.plot(T_rng, 1 - bar_alpha_linear, label="Linear", color="#00204E")
    plt.plot(T_rng, 1 - bar_alpha_cosine, label="Cosine", color="#800000")
    plt.ylabel("$1 - \\bar\\alpha_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0, 1])
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("./examples/ex_schedules.png")
    plt.show()

