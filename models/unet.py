import torch 
import torch.nn as nn

from config import t_embeddings, T_DIM

class ResidualBlock(nn.Module):
    """Residual block with timestep embedding."""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        num_groups = min(num_groups, in_ch)
        self.gn1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1) # (B, out_ch, H, W)
        self.act = nn.SiLU() 
        self.linear = nn.Linear(T_DIM, out_ch) # (B, out_ch)
        self.gn2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1) # (B, out_ch, H, W)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity() # (B, out_ch, H, W)

    def forward(self, x, t_embed):
        t_embed = self.linear(t_embed).unsqueeze(-1).unsqueeze(-1) # (B, out_ch, 1, 1)
        embeddings = self.act(self.conv1(self.gn1(x))) + t_embed # (B, out_ch, H, W)
        return self.act(self.skip(x) + self.conv2(self.gn2(embeddings)))

class ResidualBlockSeq(nn.Module):
    """Sequence of residual blocks."""
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, t_embed):
        for block in self.blocks:
            x = block(x, t_embed)
        return x

class SelfAttention(nn.Module):
    """Self-attention layer for feature maps."""
    def __init__(self, channels):
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).view(B, C, H*W).permute(0, 2, 1)  # B x N x C
        k = self.k(x).view(B, C, H*W)                     # B x C x N
        v = self.v(x).view(B, C, H*W).permute(0, 2, 1)  # B x N x C

        attn = torch.softmax(q @ k / (C**0.5), dim=-1)  # B x N x N
        out = attn @ v                                 # B x N x C
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + self.gamma * out

class UNet(nn.Module):
    """U-Net architecture for MNIST diffusion model."""
    def __init__(self):
        super().__init__()

        # --- Timestep embedding ---
        self.embed = nn.Sequential(
            nn.Linear(T_DIM, 128),
            nn.SiLU(),
            nn.Linear(128, T_DIM)
        )

        # --- Encoder ---
        self.enc1 = ResidualBlockSeq([
            ResidualBlock(1, 16),
            ResidualBlock(16, 16)
        ])
        self.down1 = nn.Conv2d(16, 32, 4, stride=2, padding=1)  # 28 → 14

        self.enc2 = ResidualBlockSeq([
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        ])
        self.down2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 14 → 7
        self.attn14 = SelfAttention(32)  # 32 channels at 14x14

        # --- Bottleneck ---
        self.bottleneck = ResidualBlockSeq([
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        ])

        # --- Decoder ---
        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 7 → 14
        self.dec2 = ResidualBlockSeq([
            ResidualBlock(64, 32),
            ResidualBlock(32, 32)
        ])

        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 14 → 28
        self.dec1 = ResidualBlockSeq([
            ResidualBlock(32, 16),
            ResidualBlock(16, 16)
        ])

        self.final = nn.Conv2d(16, 1, 1) # -> (B, 1, 28, 28)
    
    def forward(self, x, t):
        # Embed timestep
        t_embed = t_embeddings[t]  # -> (B, T_DIM)
        if t_embed.ndim == 1: 
            t_embed = t_embed.unsqueeze(0)  # now shape [B, T_DIM] with B=1
        
        t_embed = self.embed(t_embed)  # -> (B, T_DIM)

        # Encoder
        x1 = self.enc1(x, t_embed) # -> (B, 16, 28, 28)
        x2 = self.enc2(self.down1(x1), t_embed) # -> (B, 32, 14, 14)
        x2 = self.attn14(x2)  

        # Bottleneck
        x3 = self.bottleneck(self.down2(x2), t_embed) # -> (B, 64, 7, 7)

        # Decoder 
        x4 = self.up2(x3) # -> (B, 32, 14, 14)
        x4 = torch.cat([x4, x2], dim=1) # -> (B, 64, 14, 14)
        x4 = self.dec2(x4, t_embed) # -> (B, 32, 14, 14)

        x5 = self.up1(x4) # -> (B, 16, 28, 28)
        x5 = torch.cat([x5, x1], dim=1) # -> (B, 32, 28, 28)
        x5 = self.dec1(x5, t_embed) # -> (B, 16, 28, 28)

        output = self.final(x5) # -> (B, 1, 28, 28)
        return output