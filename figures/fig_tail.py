from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats


if __name__ == "__main__":

    T = 1000
    linspace = np.linspace(0, 6, T)
    tail = np.linspace(2.5, 6, T)

    plt.figure(figsize=(7, 2.0), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(linspace, sp.stats.expon.pdf(linspace), color="black", label="$f(x)$")
    plt.fill_between(tail, sp.stats.expon.pdf(tail), color="grey")
    plt.legend(loc="upper right")
    plt.yticks([])
    plt.xticks([])
    plt.ylim((0, 1))
    plt.subplot(1, 2, 2)
    plt.plot(-linspace[::-1], sp.stats.expon.pdf(linspace)[::-1], color="black", label="$f(y)$")
    plt.fill_between(-tail[::-1], sp.stats.expon.pdf(tail)[::-1], color="grey")
    plt.legend(loc="upper right")
    plt.yticks([])
    plt.xticks([])
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig("./figures/fig_tail.png")
    plt.show()
