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
sys.path.insert(0, os.path.dirname(__file__))


import torch
import numpy as np
from PIL import Image

from models.unet import UNet
from diffusion.reverse import reverse_with_visualization

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_PATH    = os.path.join(os.path.dirname(__file__), "models/mnist/model_dict/mnist_unet.pth")
CHANNELS     = 1
IMAGE_SIZE   = 28
DISPLAY_SIZE = 256   # upscale factor for the browser
SAVE_EVERY   = 100   # capture at t = 1000, 900, 800, ... 100, 0  (11 frames)

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

def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (1, C, H, W) tensor in [-1, 1] to a 256x256 RGB PIL Image.
    """
    img = tensor.squeeze(0)                       # (C, H, W)
    img = (img.clamp(-1, 1) + 1.0) / 2.0         # [0, 1]
    arr = img.permute(1, 2, 0).numpy()            # (H, W, C)

    if arr.shape[2] == 1:
        arr = arr[:, :, 0]                        # (H, W) grayscale

    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    pil = Image.fromarray(arr, mode=mode)
    return pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST).convert("RGB")

# ── Main generation function ──────────────────────────────────────────────────

def generate_mnist(model: UNet, device: torch.device = None):
    """
    Run one reverse diffusion pass and return the final image + process frames.

    Returns
    -------
    final_image : PIL.Image
        The clean generated digit at t=0, upscaled to 256x256 RGB.

    frames : list[PIL.Image]
        11 frames ordered noisy -> clean (t=1000, 900, ..., 100, 0).
        Pass directly to make_gif() in main.py.
    """
    if device is None:
        device = next(model.parameters()).device

    # Single diffusion pass — captures a frame every 100 steps
    # reverse_with_visualization returns [(tensor, t), ...] from t=999 down to t=0
    raw = reverse_with_visualization(
        model,
        device=device,
        channels=CHANNELS,
        image_size=IMAGE_SIZE,
        save_every=SAVE_EVERY,
    )

    # Sort ascending by t so index 0 = t=0 (clean), last = highest t (noisy)
    raw.sort(key=lambda x: x[1])

    # Final image is t=0
    final_image = _to_pil(raw[0][0])

    # GIF plays noisy -> clean, so reverse the sorted list
    frames = [_to_pil(tensor) for tensor, _ in reversed(raw)]

    return final_image, frames