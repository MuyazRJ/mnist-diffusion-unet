# models/unet.py
import torch
import torch.nn as nn

from config import t_embeddings, T_DIM


def _valid_groups(channels: int, max_groups: int = 8) -> int:
    """
    Pick the largest group count <= max_groups that divides channels.
    This avoids GroupNorm channel divisibility errors.
    """
    g = min(max_groups, channels)
    while g > 1:
        if channels % g == 0:
            return g
        g -= 1
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, max_groups=8):
        super().__init__()

        # Choose valid group counts for GroupNorm based on channel size
        g1 = _valid_groups(in_ch, max_groups)
        g2 = _valid_groups(out_ch, max_groups)

        self.gn1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.act = nn.SiLU()

        # Projects timestep embedding to match the current feature channel size
        self.linear = nn.Linear(T_DIM, out_ch)

        self.gn2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection uses 1x1 conv if the number of channels changes
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_embed):
        # Project timestep embedding and reshape so it can be added to feature maps
        t_proj = self.linear(t_embed).unsqueeze(-1).unsqueeze(-1)

        # First conv block with timestep conditioning
        h = self.act(self.conv1(self.gn1(x))) + t_proj

        # Second conv block
        h = self.conv2(self.gn2(h))

        # Add skip connection and apply activation
        return self.act(self.skip(x) + h)


class ResidualBlockSeq(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, t_embed):
        # Pass input through each residual block using the same timestep embedding
        for block in self.blocks:
            x = block(x, t_embed)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 1x1 convolutions used to produce query, key, and value projections
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        # Learnable scaling factor for the attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape into sequences for attention computation
        q = self.q(x).view(B, C, H * W).permute(0, 2, 1)
        k = self.k(x).view(B, C, H * W)
        v = self.v(x).view(B, C, H * W).permute(0, 2, 1)

        # Compute attention weights over spatial positions
        attn = torch.softmax(q @ k / (C ** 0.5), dim=-1)

        # Apply attention weights to value tensor
        out = attn @ v
        out = out.permute(0, 2, 1).view(B, C, H, W)

        # Residual attention output
        return x + self.gamma * out


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, max_groups=8):
        super().__init__()

        # Store timestep embedding table as a buffer so it moves with the model device
        if isinstance(t_embeddings, torch.Tensor):
            te = t_embeddings.float()
        else:
            te = t_embeddings.to("cpu").float()
        self.register_buffer("t_table", te, persistent=False)

        # Small MLP used to process timestep embeddings before injecting them into the network
        self.embed = nn.Sequential(
            nn.Linear(T_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, T_DIM),
        )

        # First encoder stage
        self.enc1 = ResidualBlockSeq([
            ResidualBlock(in_channels, 16, max_groups=max_groups),
            ResidualBlock(16, 16, max_groups=max_groups),
        ])
        self.down1 = nn.Conv2d(16, 32, 4, stride=2, padding=1)

        # Second encoder stage
        self.enc2 = ResidualBlockSeq([
            ResidualBlock(32, 32, max_groups=max_groups),
            ResidualBlock(32, 32, max_groups=max_groups),
        ])
        self.attn16 = SelfAttention(32)
        self.down2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)

        # Bottleneck stage
        self.bottleneck = ResidualBlockSeq([
            ResidualBlock(64, 64, max_groups=max_groups),
            ResidualBlock(64, 64, max_groups=max_groups),
        ])

        # First decoder stage
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec2 = ResidualBlockSeq([
            ResidualBlock(64, 32, max_groups=max_groups),
            ResidualBlock(32, 32, max_groups=max_groups),
        ])

        # Final decoder stage
        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.dec1 = ResidualBlockSeq([
            ResidualBlock(32, 16, max_groups=max_groups),
            ResidualBlock(16, 16, max_groups=max_groups),
        ])

        # Final projection back to image space
        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x, t):
        # Make sure timestep tensor is on the same device and has the correct dtype
        if isinstance(t, int):
            t = torch.tensor([t], device=x.device, dtype=torch.long)
        else:
            t = t.to(device=x.device, dtype=torch.long)

        # Look up timestep embeddings and pass them through the embedding MLP
        t_embed = self.t_table[t]
        if t_embed.ndim == 1:
            t_embed = t_embed.unsqueeze(0)

        t_embed = self.embed(t_embed)

        # Encoder path
        x1 = self.enc1(x, t_embed)
        x2 = self.enc2(self.down1(x1), t_embed)
        x2 = self.attn16(x2)

        # Bottleneck
        x3 = self.bottleneck(self.down2(x2), t_embed)

        # Decoder path with skip connections
        x4 = self.up2(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.dec2(x4, t_embed)

        x5 = self.up1(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.dec1(x5, t_embed)

        # Output predicted noise/image
        return self.final(x5)