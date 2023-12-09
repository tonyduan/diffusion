from matplotlib import pyplot as plt
import numpy as np


def c_skip(x, sigma_data):
    return sigma_data ** 2 / (np.exp(x) ** 2 + sigma_data ** 2)

def c_out(x, sigma_data):
    return np.exp(x) * sigma_data / (np.exp(x) ** 2 + sigma_data ** 2) ** 0.5


if __name__ == "__main__":

    T = 1000
    linspace = np.linspace(-8, 6, T)

    plt.figure(figsize=(7, 2.6), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(linspace, c_skip(linspace, 1.0), color="black", label="$\\sigma_\\text{data}=1.0$")
    plt.plot(linspace, c_skip(linspace, 0.5), color="red", label="$\\sigma_\\text{data}=0.5$")
    plt.plot(linspace, c_skip(linspace, 0.25), color="blue", label="$\\sigma_\\text{data}=0.25$")
    plt.legend(loc="upper left")
    plt.xlabel("$\\log\\sigma$")
    plt.title("$c_\\text{skip}$")
    plt.tick_params(bottom=False, top=False)
    plt.subplot(1, 2, 2)
    plt.plot(linspace, c_out(linspace, 1.0), color="black", label="$\\sigma_\\text{data}=1.0$")
    plt.plot(linspace, c_out(linspace, 0.5), color="red", label="$\\sigma_\\text{data}=0.5$")
    plt.plot(linspace, c_out(linspace, 0.25), color="blue", label="$\\sigma_\\text{data}=0.25$")
    plt.legend(loc="upper left")
    plt.xlabel("$\\log\\sigma$")
    plt.title("$c_\\text{out}$")
    plt.tick_params(bottom=False, top=False)
    plt.tight_layout()
    plt.savefig("./figures/fig_karras_param.png")
    plt.show()

