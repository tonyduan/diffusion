from argparse import ArgumentParser
import logging

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim

from src.blocks import UNet
from src.models import DiffusionModel, DiffusionModelConfig


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=512, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load data from https://www.openml.org/d/554
    # (70000, 784) values between 0-255
    x, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)

    # Reshape to 32x32
    x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
    x = np.pad(x, pad_width=((0, 0), (2, 2), (2, 2)))
    x = rearrange(x, "b h w -> b (h w)")

    # Standardize to [-1, 1]
    input_mean = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    input_sd = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    x = ((x - input_mean) / input_sd).astype(np.float32)

    nn_module = UNet(1, 128, (1, 2, 4, 8))
    model = DiffusionModel(
        nn_module=nn_module,
        input_shape=(1, 32, 32,),
        config=DiffusionModelConfig(
            num_timesteps=500,
            target_type="pred_x_0",
            gamma_type="ddim",
            noise_schedule_type="cosine",
        ),
    )
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    for i in range(args.iterations):
        x_batch = x[np.random.choice(len(x), args.batch_size)]
        x_batch = torch.from_numpy(x_batch).to(args.device)
        x_batch = rearrange(x_batch, "b (h w) -> b () h w", h=32, w=32)
        optimizer.zero_grad()
        loss = model.loss(x_batch).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")

    model.eval()

    samples = model.sample(bsz=64, num_sampling_timesteps=None, device=args.device).cpu().numpy()
    samples = rearrange(samples, "t b () h w -> t b (h w)")
    samples = samples * input_sd + input_mean
    x_vis = x[:64] * input_sd + input_mean

    nrows, ncols = 10, 2
    raster = np.zeros((nrows * 32, ncols * 32 * 6))
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), 32 * col : 32 * (col + 1)] = x_vis[i].reshape(32, 32)
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), 32 * ncols + 32 * col : 32 * ncols + 32 * (col + 1)] = samples[500][i].reshape(32, 32)
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), 32 * ncols * 2 + 32 * col : 32 * ncols * 2 + 32 * (col + 1)] = samples[375][i].reshape(32, 32)
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), 32 * ncols * 3 + 32 * col : 32 * ncols * 3 + 32 * (col + 1)] = samples[250][i].reshape(32, 32)
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), 32 * ncols * 4 + 32 * col : 32 * ncols * 4 + 32 * (col + 1)] = samples[125][i].reshape(32, 32)
    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), 32 * ncols * 5 + 32 * col : 32 * ncols * 5 + 32 * (col + 1)] = samples[0][i].reshape(32, 32)

    plt.imsave("./examples/ex_mnist.png", raster, vmin=0, vmax=255)
