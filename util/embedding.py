import math
import torch

def get_timestep_embedding(t, dim):
    """
    t: tensor of shape (B,) containing timesteps 0..T-1
    dim: embedding dimension
    returns: tensor of shape (B, dim)
    """
    half_dim = dim // 2
    # Compute frequencies
    freq = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
    angles = t[:, None].float() * freq[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return emb  # (B, dim)
