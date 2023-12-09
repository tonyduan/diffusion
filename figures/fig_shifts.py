import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt

from figures.fig_schedules import get_cosine_schedule


def draw_lines(logsnr, color="black", stride=10):
    alpha = expit(logsnr) ** 0.5
    sigma = expit(-logsnr) ** 0.5
    for x, y in zip(alpha[::stride], sigma[::stride]):
        plt.plot([0, x], [0, y], color=color, alpha=0.5)


if __name__ == "__main__":

    T = 201
    T_rng = np.linspace(0, 1, T + 1)

    alpha = get_cosine_schedule(T)
    sigma = (1 - alpha ** 2) ** 0.5
    cosine_logsnr = 2 * (np.log(alpha) - np.log(sigma))

    shifted_down_logsnr = cosine_logsnr + 2 * np.log(0.5)
    shifted_up_logsnr = cosine_logsnr + 2 * np.log(2)

    plt.figure(figsize=(7, 5.2), dpi=200)

    plt.subplot(2, 3, 1)
    plt.plot(T_rng, cosine_logsnr, label="Linear", color="black")
    plt.plot(T_rng, shifted_down_logsnr, label="Shifted down", color="navy")
    plt.plot(T_rng, shifted_up_logsnr, label="Shifted up", color="maroon")
    plt.ylabel("$\\log SNR_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([-10, 10])
    plt.ylim([-11, 11])
    plt.legend(loc="lower left")

    plt.subplot(2, 3, 2)
    plt.plot(T_rng, expit(cosine_logsnr) ** 0.5, label="Linear", color="black")
    plt.plot(T_rng, expit(shifted_down_logsnr) ** 0.5, label="Shifted down", color="navy")
    plt.plot(T_rng, expit(shifted_up_logsnr) ** 0.5, label="Shifted up", color="maroon")
    plt.ylabel("$\\alpha_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0, 1])
    plt.legend(loc="lower left")

    plt.subplot(2, 3, 3)
    plt.plot(T_rng, expit(-cosine_logsnr) ** 0.5, label="Linear", color="black")
    plt.plot(T_rng, expit(-shifted_down_logsnr) ** 0.5, label="Shifted down", color="navy")
    plt.plot(T_rng, expit(-shifted_up_logsnr) ** 0.5, label="Shifted up", color="maroon")
    plt.ylabel("$\\sigma_t$")
    plt.xlabel("Timestep $t$")
    plt.xticks([])
    plt.yticks([0, 1])
    plt.legend(loc="lower left")

    plt.subplot(2, 3, 4)
    draw_lines(cosine_logsnr)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$x_t$")
    plt.ylabel("$\\epsilon_t$")
    plt.title("Cosine Schedule")

    plt.subplot(2, 3, 5)
    draw_lines(shifted_down_logsnr, color="navy")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$x_t$")
    plt.ylabel("$\\epsilon_t$")
    plt.title("Shifted down $+2\\log\\left(\\frac{1}{2}\\right)$")

    plt.subplot(2, 3, 6)
    draw_lines(shifted_up_logsnr, color="maroon")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("$x_t$")
    plt.ylabel("$\\epsilon_t$")
    plt.title("Shifted up $+2\\log(2)$")

    plt.tight_layout()
    plt.savefig("./figures/fig_shifts.png")

