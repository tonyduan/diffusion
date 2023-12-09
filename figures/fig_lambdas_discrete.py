import numpy as np
from matplotlib import pyplot as plt

from figures.fig_schedules import get_cosine_schedule


if __name__ == "__main__":

    T = 15
    linspace = np.linspace(-7, 7, 500)

    plt.figure(figsize=(7, 2.6), dpi=300)

    #
    # First plot the cosine schedule
    #
    alpha = get_cosine_schedule(T)
    sigma = (1 - alpha ** 2) ** 0.5

    log_lambd = np.log(sigma) - np.log(alpha)

    for idx, one_lambd in enumerate(log_lambd):
        label = "Cosine Schedule" if idx == 0 else None
        plt.axvline(one_lambd, color="black", linestyle="--", label=label, alpha=0.5)

    #
    # Second plot the score matching geometric distribution
    #
    min_sigma = 0.002
    max_sigma = 80.0
    log_lambd = np.linspace(np.log(min_sigma), np.log(max_sigma), T)

    for idx, one_lambd in enumerate(log_lambd):
        label = "Score Matching" if idx == 0 else None
        plt.axvline(one_lambd, color="maroon", linestyle="--", label=label, alpha=0.5)

    plt.xlim((-7, 7))
    plt.yticks([])
    plt.xlabel("$\\log\\lambda$")
    plt.legend()
    plt.tight_layout()
    plt.tick_params(bottom=False, top=False)
    plt.savefig("./figures/fig_lambdas_discrete.png")

