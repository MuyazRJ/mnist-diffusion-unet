# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#

# load.py
import argparse
import math

import torch
import matplotlib.pyplot as plt

from models.unet import UNet
from diffusion.reverse import reverse


def main():
    # Set up command-line arguments for checkpoint, dataset, sample count, and device
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="model_dict/mnist_unet.pth")
    ap.add_argument("--dataset", type=str, choices=["mnist", "cifar"], default="mnist",
                    help="mnist -> (1,28,28), cifar -> (3,32,32)")
    ap.add_argument("--n", type=int, default=50, help="How many images to sample")
    ap.add_argument("--cols", type=int, default=10, help="Grid columns")
    ap.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu (default: auto)")
    args = ap.parse_args()

    # Choose the requested device, or automatically pick CUDA if available
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set image shape based on the chosen dataset
    if args.dataset == "mnist":
        channels, image_size = 1, 28
        cmap = "gray"
    else:
        channels, image_size = 3, 32
        cmap = None

    # Build a U-Net with the correct number of input and output channels
    model = UNet(in_channels=channels, out_channels=channels).to(device)

    # Load the trained checkpoint
    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.ckpt, map_location=device)

    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded: {args.ckpt}")

    # Generate a batch of images using the reverse diffusion process
    imgs = reverse(
        model,
        device=device,
        batch_size=args.n,
        channels=channels,
        image_size=image_size,
    )

    imgs = imgs.cpu()

    # Set up the plot grid dimensions
    cols = args.cols
    rows = math.ceil(args.n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1:
        axes = [axes]

    # Display sampled images in a grid
    idx = 0
    for r in range(rows):
        row_axes = axes[r] if rows > 1 else axes
        for c in range(cols):
            ax = row_axes[c] if rows > 1 else row_axes[c]
            ax.axis("off")

            if idx >= args.n:
                continue

            img = imgs[idx]
            if channels == 1:
                ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()