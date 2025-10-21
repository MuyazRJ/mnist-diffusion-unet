import torch

from config import alpha_bars


def forward(x0, t):
    """
    Adds noise to the input image x0 at timestep t using the forward diffusion process.
    Args:
        x0 (torch.Tensor): Original image tensor of shape [C, H, W] or [1, C, H, W]
        t (int): Timestep at which to add noise (1 <= t <= T)
    Returns:    
        noisy_image (torch.Tensor): Noisy image at timestep t
        eps (torch.Tensor): The noise added to the original image
    """
    alpha_bar_t = alpha_bars[t - 1].view(-1, 1, 1, 1)  # Reshape for broadcasting
    eps = torch.randn_like(x0)
    noisy_image = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    return noisy_image, eps
    