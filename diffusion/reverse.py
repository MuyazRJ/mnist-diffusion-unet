import torch

from config import T, alpha_bars, alphas, betas

import matplotlib.pyplot as plt
from math import ceil

def reverse(model):
    x_t = torch.randn((1, 1, 28, 28))  # Start from pure noise
    for t in reversed(range(0, T)):
        with torch.no_grad():
            eps_pred = model(x_t, t)
        
        x_t = (1/torch.sqrt(alphas[t])) * (x_t - ((1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t])) * eps_pred)

        if t > 0:
            z = torch.randn_like(x_t)
            x_t += torch.sqrt(betas[t]) * z
    
    x_t = x_t.clamp(-1, 1)
    x_img = (x_t + 1) / 2.0
    x_img_255 = (x_img * 255).byte()
    return x_img_255

