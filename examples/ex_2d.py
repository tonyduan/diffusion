from argparse import ArgumentParser
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.blocks import FFN
from src.simple.diffusion import DiffusionModel, DiffusionModelConfig


def gen_mixture_data(n=512):
    x = np.r_[np.random.randn(n // 2, 2) + np.array([-5, 0]),
              np.random.randn(n // 2, 2) + np.array([5, 0])]
    x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, ddof=1, keepdims=True)
    return x.astype(np.float32)

def plot_data(x):
    plt.hist2d(x[:,0].numpy(), x[:,1].numpy(), bins=100, range=np.array([(-3, 3), (-6, 6)]))
    plt.axis("off")


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--iterations", default=2000, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    nn_module = FFN(in_dim=2, embed_dim=16)
    model = DiffusionModel(
        nn_module=nn_module,
        input_shape=(2,),
        config=DiffusionModelConfig(
            num_timesteps=10,
            noise_schedule_type="linear",
            target_type="pred_eps",
            gamma_type="ddpm",
        ),
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    for i in range(args.iterations):
        x = torch.from_numpy(gen_mixture_data(args.n))
        optimizer.zero_grad()
        loss = model.loss(x).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")

    model.eval()
    samples = model.sample(bsz=512, device="cpu")

    plt.figure(figsize=(12, 12))
    plt.subplot(4, 4, 1)
    plot_data(x)
    plt.title("Actual data")
    for t in range(11):
        plt.subplot(4, 4, t + 2)
        plot_data(samples[t])
        plt.title(f"Sample t={t}")

    samples = model.sample(bsz=512, device="cpu", num_sampling_timesteps=3)

    for t in range(4):
        plt.subplot(4, 4, t + 12 + 1)
        plot_data(samples[t])
        plt.title(f"Accelerated Sample t={t}")

    plt.tight_layout()
    plt.savefig("./examples/ex_2d.png")

    x_grid, y_grid = torch.meshgrid(torch.linspace(-3, 3, 30), torch.linspace(-6, 6, 20), indexing="ij")
    x = torch.from_numpy(np.c_[x_grid.flatten(), y_grid.flatten()])
    bsz, _ = x.shape

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes[0, 0].axis("off")

    for scalar_t in range(0, model.num_timesteps + 1):

        t = torch.full((bsz,), fill_value=scalar_t, device=x.device)

        with torch.no_grad():
            pred_eps = model.nn_module(x, t)

        pred_eps = pred_eps.numpy().reshape(30, 20, 2)
        one_axes = axes[(scalar_t + 1) // 4, (scalar_t + 1) % 4]
        one_axes.quiver(x_grid.numpy(), y_grid.numpy(), -pred_eps[..., 0], -pred_eps[..., 1])
        one_axes.axis("off")
        one_axes.set_title(f"Score {scalar_t}")

    fig.tight_layout()
    fig.savefig("./examples/ex_2d_quiver.png")

