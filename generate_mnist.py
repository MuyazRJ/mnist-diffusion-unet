# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#

"""
generate_mnist.py
-----------------
Loads the trained MNIST DDPM and produces:
  - final_image : PIL.Image  (upscaled to 256x256, RGB)
  - frames      : list[PIL.Image]  (11 RGB frames, t=1000 down to t=0 in
                  steps of 100, ordered noisy -> clean for the GIF)

Usage in main.py
----------------
    from generate_mnist import load_mnist_model, generate_mnist

    mnist_model = load_mnist_model()          # once at startup
    img, frames = generate_mnist(mnist_model) # once per request
"""

import sys, os

_MNIST_ROOT = os.path.dirname(__file__)
if _MNIST_ROOT not in sys.path:
    sys.path.insert(0, _MNIST_ROOT)

for _mod in list(sys.modules.keys()):
    if _mod in ("config", "model", "diffusion", "utils") or \
       _mod.startswith(("model.", "diffusion.", "utils.", "models.", "data_loaders.")):
        del sys.modules[_mod]

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.unet import UNet
from diffusion.reverse import reverse_with_visualization

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_PATH    = os.path.join(os.path.dirname(__file__), "model_dict/mnist_unet.pth")
CHANNELS     = 1
IMAGE_SIZE   = 28
DISPLAY_SIZE = 256
SAVE_EVERY   = 100   # captures t = 1000, 900, 800, ... 100, 0  (11 frames)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_mnist_model(ckpt_path: str = CKPT_PATH, device: torch.device = None) -> UNet:
    """Load and return the UNet. Call once at server startup."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=CHANNELS, out_channels=CHANNELS).to(device)

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state)
    model.eval()
    print(f"[MNIST] Loaded weights from {ckpt_path} on {device}")
    return model

# ── Helper: tensor -> PIL ─────────────────────────────────────────────────────

def _to_pil(tensor: torch.Tensor, timestep: int = None) -> Image.Image:
    """
    Convert a (1, C, H, W) tensor in [-1, 1] to a 256x256 RGB PIL Image.
    Optionally stamps the timestep label in the top-left corner.
    """
    img = tensor.squeeze(0)
    img = (img.clamp(-1, 1) + 1.0) / 2.0
    arr = img.permute(1, 2, 0).numpy()

    if arr.shape[2] == 1:
        arr = arr[:, :, 0]

    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    pil = Image.fromarray(arr, mode=mode)
    pil = pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST).convert("RGB")

    if timestep is not None:
        draw = ImageDraw.Draw(pil)
        label = f"t = {timestep}"
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except IOError:
            font = ImageFont.load_default()
        draw.text((11, 11), label, fill=(0, 0, 0), font=font)
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    return pil

# ── Main generation function ──────────────────────────────────────────────────

def generate_mnist(model: UNet, device: torch.device = None):
    """
    Run one reverse diffusion pass and return the final image + process frames.

    Returns
    -------
    final_image : PIL.Image
        The clean generated digit at t=0, upscaled to 256x256 RGB.

    frames : list[PIL.Image]
        11 frames ordered noisy -> clean (t=1000, 900, ..., 100, 0),
        each with the timestep stamped in the top-left corner.
    """
    if device is None:
        device = next(model.parameters()).device

    raw = reverse_with_visualization(
        model,
        device=device,
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        save_every=SAVE_EVERY,
    )

    # Sort ascending: index 0 = t=0 (cleanest)
    raw.sort(key=lambda x: x[1])

    # Final image at t=0, no label
    final_image = _to_pil(raw[0][0])

    # GIF plays noisy -> clean (reversed), with timestep stamped on each frame
    frames = [_to_pil(tensor, timestep=t) for tensor, t in reversed(raw)]

    return final_image, frames