from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats


def cosine_pdf(x, kappa):
    #
    # Equivalent to hyperbolic secant distribution (ignore truncation due to sigma_min, sigma_max).
    #
    return 1 / np.pi / np.cosh(x + np.log(kappa))

def normal_pdf(x, loc, scale, sigma_min, sigma_max):
    #
    # Simple truncated normal.
    #
    return sp.stats.truncnorm.pdf(x, a=np.log(sigma_min), b=np.log(sigma_max), loc=loc, scale=scale)

def exp_pdf(x, sigma_min, sigma_max, rho):
    #
    # Adjust for the truncation due to [sigma_min, sigma_max] using partition constant.
    #
    partition_constant = 1 - np.exp(-(np.log(sigma_max) - np.log(sigma_min)) / rho)
    return 1 / rho * np.exp((x - np.log(sigma_max)) / rho) / partition_constant


if __name__ == "__main__":

    T = 1000
    linspace = np.linspace(-8, 6, T)

    plt.figure(figsize=(7, 2.6), dpi=300)
    plt.plot(linspace, cosine_pdf(linspace, 1.0), color="black", label="Cosine($\\kappa=1.0$)")
    plt.plot(linspace, cosine_pdf(linspace, 0.5), color="red", label="Cosine($\\kappa=0.5$)")
    plt.plot(linspace, normal_pdf(linspace, -1.2, 1.2, 0.002, 80), color="green", label="N($\\mu=-1.2, \\gamma=1.2$)")
    plt.plot(linspace, exp_pdf(linspace, 0.002, 80, 7), color="blue", label="Exp($\\rho=7$)")
    plt.legend()
    plt.yticks([])
    plt.xlabel("$\\log\\sigma$")
    plt.tight_layout()
    plt.tick_params(bottom=False, top=False)
    plt.savefig("./figures/fig_lambdas_continuous.png")
    plt.show()

