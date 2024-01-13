from argparse import ArgumentParser
import logging

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.blocks_3d import UNet3d
from src.schedules import CosineSchedule
from src.score_matching import ScoreMatchingModel, ScoreMatchingModelConfig


class MovingMNISTDataset(Dataset):

    # Standardize to [-1, 1]
    input_mean = 127.5
    input_sd = 127.5

    def __init__(self, num_frames, height, width, velocity=5):

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.velocity = velocity
        self.mnist_L = 28  # 28 x 28 images

        # Load data from https://www.openml.org/d/554
        # (70000, 784) values between 0-255
        x, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)

        x = ((x - MovingMNISTDataset.input_mean) / MovingMNISTDataset.input_sd).astype(np.float32)
        self.x = rearrange(x, "b (h w) -> b h w", h=28, w=28)

        # Enumerate all possible indices in this dataset, we allow 4 directions
        self.dir_to_dy_dx = {
            0: (-self.velocity, 0),
            1: (self.velocity, 0),
            2: (0, -self.velocity),
            3: (0, self.velocity),
        }
        self.index_shape = (len(self.dir_to_dy_dx), self.height - self.mnist_L, self.width - self.mnist_L, len(self.x))

    def __len__(self):
        return np.prod(self.index_shape)

    def __getitem__(self, idx):

        result = np.full((self.num_frames, self.height, self.width), dtype=np.float32, fill_value=-1)
        dir_idx, y_idx, x_idx, mnist_idx = np.unravel_index(idx, self.index_shape)

        R = self.mnist_L // 2

        # Denotes center coords
        y, x = y_idx + R, x_idx + R
        dy, dx = self.dir_to_dy_dx[dir_idx]

        for t in range(self.num_frames):
            result[t, y - R : y + R, x - R : x + R] = self.x[mnist_idx]
            if y + dy - R <= 0 or y + dy + R >= self.height:
                dy *= -1
            if x + dx - R <= 0 or x + dx + R >= self.width:
                dx *= -1
            y += dy
            x += dx

        # Draw boundaries along corner of image
        result[:, :, 0].fill(1.0)
        result[:, :, -1].fill(1.0)
        result[:, 0, :].fill(1.0)
        result[:, -1, :].fill(1.0)

        return result[np.newaxis]

    @staticmethod
    def visualize_one(x):
        num_channels, num_frames, height, width = x.shape
        assert num_channels == 1
        x = x * MovingMNISTDataset.input_sd + MovingMNISTDataset.input_mean
        x = np.clip(x, a_min=0, a_max=255)
        x = x.astype(np.uint8)
        raster = np.zeros((height, width * num_frames), dtype=np.uint8)
        for t in range(num_frames):
            raster[:, width * t : width * (t + 1)] = x[:, t].squeeze(axis=0)
        return raster


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=16, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=0, type=int, choices=(0, 1))
    argparser.add_argument("--num-frames", default=4, type=int)
    argparser.add_argument("--velocity", default=4, type=int)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    nn_module = UNet3d(1, 128, (1, 2, 4, 8))
    model = ScoreMatchingModel(
        nn_module=nn_module,
        input_shape=(1, args.num_frames, 64, 64),
        config=ScoreMatchingModelConfig(
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
            train_sigma_schedule=CosineSchedule(kappa=0.25),
            test_sigma_schedule=CosineSchedule(kappa=0.25),
        ),
    )
    model = model.to(args.device)

    dataset = MovingMNISTDataset(args.num_frames, 64, 64, velocity=args.velocity)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    if args.load_trained:
        model.load_state_dict(torch.load("./ckpts/moving_mnist_trained.pt"))
    else:
        iterator = iter(dataloader)
        for step_num in range(args.iterations):
            try:
                x_batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                x_batch = next(iterator)
            x_batch = x_batch.to(args.device)
            optimizer.zero_grad()
            loss = model.loss(x_batch).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step_num % 100 == 0:
                logger.info(f"Iter: {step_num}\t" + f"Loss: {loss.data:.2f}\t")
        torch.save(model.state_dict(), "./ckpts/moving_mnist_trained.pt")

    model.eval()

    NUM_SHOWN = 4

    samples = model.sample(bsz=NUM_SHOWN, num_sampling_timesteps=20, device=args.device).cpu().numpy()
    samples = samples[0]

    gt_raster = np.concatenate([
        MovingMNISTDataset.visualize_one(dataset[i]) for i in range(NUM_SHOWN)
    ], axis=0)
    pred_raster = np.concatenate([
        MovingMNISTDataset.visualize_one(samples[i]) for i in range(NUM_SHOWN)
    ], axis=0)

    raster = np.concatenate([gt_raster, pred_raster], axis=0)
    plt.imsave("./examples/ex_moving_mnist.png", raster, vmin=0, vmax=255)
