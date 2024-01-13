from argparse import ArgumentParser
import logging

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.blocks import UNet
from src.score_matching import ScoreMatchingModel, ScoreMatchingModelConfig


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=0, type=int, choices=(0, 1))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    nn_module = UNet(3, 128, (1, 2, 4, 8))
    model = ScoreMatchingModel(
        nn_module=nn_module,
        input_shape=(3, 32, 32,),
        config=ScoreMatchingModelConfig(
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
        ),
    )
    model = model.to(args.device)

    # Standardize to [-1, +1]
    dataset_mean, dataset_sd = np.asarray([0.5, 0.5, 0.5]), np.asarray([0.5, 0.5, 0.5])
    dataset = datasets.CIFAR10(
        "./data/cifar_10",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_sd),
        ])
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.load_trained:
        model.load_state_dict(torch.load("./ckpts/cifar_trained.pt"))
    else:
        iterator = iter(dataloader)
        for step_num in range(args.iterations):
            try:
                x_batch, _ = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                x_batch, _ = next(iterator)
            x_batch = x_batch.to(args.device)
            optimizer.zero_grad()
            loss = model.loss(x_batch).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step_num % 100 == 0:
                logger.info(f"Iter: {step_num}\t" + f"Loss: {loss.data:.2f}\t")
        torch.save(model.state_dict(), "./ckpts/cifar_trained.pt")

    model.eval()

    samples = model.sample(bsz=64, num_sampling_timesteps=20, device=args.device).cpu().numpy()
    samples = samples * rearrange(dataset_sd, "c -> 1 1 c 1 1") + rearrange(dataset_mean, "c -> 1 1 c 1 1")
    samples = np.rint(samples * 255).clip(min=0, max=255).astype(np.uint8)

    x_vis = rearrange(dataset.data[:64], "b h w c -> b c h w")

    nrows, ncols = 10, 2
    percents = (100, 75, 50, 25, 0)
    raster = np.zeros((3, nrows * 32, ncols * 32 * (len(percents) + 1)), dtype=np.uint8)

    for i in range(nrows * ncols):
        row, col = i // ncols, i % ncols
        raster[:, 32 * row : 32 * (row + 1), 32 * col : 32 * (col + 1)] = x_vis[i]
    for percent_idx, percent in enumerate(percents):
        itr_num = int(round(0.01 * percent * (len(samples) - 1)))
        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            offset = 32 * ncols * (percent_idx + 1)
            raster[:, 32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = samples[itr_num][i]

    raster = rearrange(raster, "c h w -> h w c")
    plt.imsave("./examples/ex_cifar.png", raster, vmin=0, vmax=255)
