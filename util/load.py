from models.unet import UNet  # or wherever your model is defined
import torch
import matplotlib.pyplot as plt

model = UNet()  # recreate same architecture
model.load_state_dict(torch.load("model_dict/new.pth", map_location="cpu"))
model.eval()

from diffusion.reverse import reverse, reverse_with_visualization

import math

n = 50
cols = 10  # number of images per row
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

for i in range(n):
    img = reverse(model)
    r, c = divmod(i, cols)
    axes[r, c].imshow(img.squeeze().cpu(), cmap='gray')
    axes[r, c].axis('off')

# Hide any empty subplots
for j in range(n, rows * cols):
    r, c = divmod(j, cols)
    axes[r, c].axis('off')

plt.tight_layout()
plt.show()

#reverse_with_visualization(model)