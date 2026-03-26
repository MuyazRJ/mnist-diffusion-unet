# diffusion/reverse.py
import torch
from config import T, alpha_bars, alphas, betas

# Cache schedules per device so we don't keep copying
_SCHED_CACHE = {}

def _get_schedules(device: torch.device):
    key = str(device)
    if key not in _SCHED_CACHE:
        _SCHED_CACHE[key] = (
            alphas.to(device).float(),
            alpha_bars.to(device).float(),
            betas.to(device).float(),
        )
    return _SCHED_CACHE[key]


@torch.inference_mode()
def reverse(
    model,
    device=None,
    batch_size: int = 1,
    channels: int = 1,
    image_size: int = 28,
):
    """
    Generic reverse sampler.
    Returns: (B, C, H, W) float in [0,1]
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    a, ab, b = _get_schedules(device)

    # Start from pure noise with the right shape
    x_t = torch.randn((batch_size, channels, image_size, image_size), device=device)

    for t in reversed(range(T)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_tensor)

        x_t = (1.0 / torch.sqrt(a[t])) * (x_t - ((1.0 - a[t]) / torch.sqrt(1.0 - ab[t])) * eps_pred)

        if t > 0:
            x_t = x_t + torch.sqrt(b[t]) * torch.randn_like(x_t)

    x_t = x_t.clamp(-1, 1)
    return (x_t + 1.0) / 2.0


def reverse_with_visualization(
    model,
    device=None,
    channels: int = 1,
    image_size: int = 28,
    save_every: int = 50,
):
    """
    Same sampler but returns intermediate frames (for plotting/debugging).
    Returns: list of (img_cpu, t) where img_cpu is (1,C,H,W) float in [-1,1]
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    a, ab, b = _get_schedules(device)

    x_t = torch.randn((1, channels, image_size, image_size), device=device)
    frames = []

    with torch.inference_mode():
        for t in reversed(range(T)):
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            eps_pred = model(x_t, t_tensor)

            x_t = (1.0 / torch.sqrt(a[t])) * (x_t - ((1.0 - a[t]) / torch.sqrt(1.0 - ab[t])) * eps_pred)

            if t > 0:
                x_t = x_t + torch.sqrt(b[t]) * torch.randn_like(x_t)

            if t % save_every == 0 or t == 0:
                frames.append((x_t.detach().cpu().clone(), t))

    return frames
