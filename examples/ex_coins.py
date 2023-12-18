from argparse import ArgumentParser
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.blocks import FFN
from src.simple.diffusion import DiffusionModel, DiffusionModelConfig


def gen_data(n=512, d=48):
    x = np.vstack([
        np.random.permutation(np.r_[np.ones(3 * d // 4), np.zeros(d // 4)])
        for _ in range(n)
    ])
    assert np.all(np.sum(x, axis=1) == 3 * d // 4)
    x = x * 2 - 1
    return x.astype(np.float32)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--iterations", default=2000, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    nn_module = FFN(in_dim=48, embed_dim=96)
    model = DiffusionModel(
        nn_module=nn_module,
        input_shape=(48,),
        config=DiffusionModelConfig(
            num_timesteps=100,
            target_type="pred_x_0",
            gamma_type="ddim",
            noise_schedule_type="cosine",
        ),
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    for i in range(args.iterations):
        x = torch.from_numpy(gen_data(args.n))
        optimizer.zero_grad()
        loss = model.loss(x).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")

    model.eval()
    samples = model.sample(bsz=512, device="cpu")
    num_heads = (samples[0] > 0).sum(dim=1).cpu().numpy()

    plt.figure(figsize=(8, 3))
    plt.hist(num_heads, range=(10, 50), bins=20, alpha=0.5, color="grey", label="Samples")
    logger.info(f"Samples mean {np.mean(num_heads):.2f} sd {np.std(num_heads):.2f}")

    random_sample = np.random.rand(args.n, 48) > 0.5
    num_heads = random_sample.sum(axis=1)

    plt.hist(num_heads, range=(10, 50), bins=20, alpha=0.5, color="black", label="Prior")
    logger.info(f"Prior mean {np.mean(num_heads):.2f} sd {np.std(num_heads):.2f}")

    plt.tight_layout()
    plt.savefig("examples/ex_coins.png")
