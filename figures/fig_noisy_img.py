import numpy as np
from matplotlib import pyplot as plt
from skimage import data


def add_noise(img, scale):
    return np.clip(img + scale * np.random.randn(*img.shape), a_min=-1, a_max=1)


if __name__ == "__main__":

    figure = plt.figure(figsize=(7, 2.6), dpi=300)

    img = data.camera()
    img = 2 * (img.astype(np.float32) / 255 - 0.5)
    assert img.shape == (512, 512)

    img_1x = img
    img_2x = img[::2, ::2]
    img_4x = img[::4, ::4]

    img_1x = add_noise(img_1x, 0.25)
    img_2x = add_noise(img_2x, 0.25)
    img_4x = add_noise(img_4x, 0.25)

    plt.subplot(1, 3, 1)
    plt.imshow(img_1x, cmap="gray")
    plt.title("512 x 512")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(img_2x, cmap="gray")
    plt.title("256 x 256")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(img_4x, cmap="gray")
    plt.title("128 x 128")
    plt.axis("off")
    plt.savefig("./figures/fig_noisy_img.png")
    plt.show()
