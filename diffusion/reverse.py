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

def reverse_with_visualization(model, device="cpu"):
    model.eval()
    x_t = torch.randn((1, 1, 28, 28), device=device)
    imgs = []

    for t in reversed(range(0, T)):
        t_tensor = torch.tensor([t], device=device)
        with torch.no_grad():
            eps_pred = model(x_t, t_tensor)

        # Reverse diffusion step
        x_t = (1/torch.sqrt(alphas[t])) * (x_t - ((1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t])) * eps_pred)

        if t > 0:
            z = torch.randn_like(x_t)
            x_t += torch.sqrt(betas[t]) * z

        # Save every 50 steps, including final
        if t % 50 == 0 or t == 0:
            imgs.append((x_t.detach().cpu().clone(), t))

    # Plot results in rows
    n = len(imgs)
    cols = 5  # number of images per row
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i, (img, t) in enumerate(imgs):
        r, c = divmod(i, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        img = img.clamp(-1, 1)
        img = (img + 1) / 2.0
        ax.imshow(img[0, 0], cmap='gray')
        ax.set_title(f"t={t}")
        ax.axis("off")

    # Hide any empty subplots
    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    final_img = imgs[-1][0].clamp(-1, 1)
    final_img = (final_img + 1) / 2.0
    return (final_img * 255).byte()