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