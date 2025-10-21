import torch 
import torch.nn as nn

from config import t_embeddings, T_DIM

class ResidualBlock(nn.Module):
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
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x, t_embed):
        for block in self.blocks:
            x = block(x, t_embed)
        return x

class SelfAttention(nn.Module):
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