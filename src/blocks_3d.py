from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.blocks import unsqueeze_as, PositionalEmbedding, SelfAttention2d


class BasicBlock3d(nn.Module):
    """
    BasicBlock: two 3x3 convs followed by a residual connection then ReLU.
    [He et al. CVPR 2016]

        BasicBlock(x) = ReLU( x + Conv3x3( ReLU( Conv3x3(x) ) ) )

    This version supports an additive shift parameterized by time.
    """
    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_c, time_c),
            nn.ReLU(),
            nn.Linear(time_c, out_c),
        )
        if in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(out_c)
            )

    def forward(self, x, t):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + unsqueeze_as(self.mlp_time(t), x))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + self.shortcut(x))
        return out


class UNet3d(nn.Module):
    """
    Simple implementation that closely mimics the one by Phil Wang (lucidrains).
    """
    def __init__(self, in_dim, embed_dim, dim_scales):
        super().__init__()

        self.init_embed = nn.Conv3d(in_dim, embed_dim, 1)
        self.time_embed = PositionalEmbedding(embed_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Example:
        #   in_dim=1, embed_dim=32, dim_scales=(1, 2, 4, 8) => all_dims=(32, 32, 64, 128, 256)
        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        for idx, (in_c, out_c) in enumerate(zip(
            all_dims[:-1],
            all_dims[1:],
        )):
            is_last = idx == len(all_dims) - 2
            self.down_blocks.extend(nn.ModuleList([
                BasicBlock3d(in_c, in_c, embed_dim),
                BasicBlock3d(in_c, in_c, embed_dim),
                nn.Conv3d(in_c, out_c, (1, 3, 3), (1, 2, 2), (0, 1, 1)) if not is_last else nn.Conv3d(in_c, out_c, 1),
            ]))

        for idx, (in_c, out_c, skip_c) in enumerate(zip(
            all_dims[::-1][:-1],
            all_dims[::-1][1:],
            all_dims[:-1][::-1],
        )):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(nn.ModuleList([
                BasicBlock3d(in_c + skip_c, in_c, embed_dim),
                BasicBlock3d(in_c + skip_c, in_c, embed_dim),
                nn.ConvTranspose3d(in_c, out_c, (1, 2, 2), (1, 2, 2)) if not is_last else nn.Conv3d(in_c, out_c, 1),
            ]))

        self.mid_blocks = nn.ModuleList([
            BasicBlock3d(all_dims[-1], all_dims[-1], embed_dim),
            SelfAttention2d(all_dims[-1]),
            BasicBlock3d(all_dims[-1], all_dims[-1], embed_dim),
        ])
        self.out_blocks = nn.ModuleList([
            BasicBlock3d(embed_dim, embed_dim, embed_dim),
            nn.Conv3d(embed_dim, in_dim, 1, bias=True),
        ])

    def forward(self, x, t):
        _, _, num_frames, *_ = x.shape
        x = self.init_embed(x)
        t = self.time_embed(t)
        skip_conns = []
        residual = x.clone()

        for block in self.down_blocks:
            if isinstance(block, BasicBlock3d):
                x = block(x, t)
                skip_conns.append(x)
            else:
                x = block(x)
        for block in self.mid_blocks:
            if isinstance(block, BasicBlock3d):
                x = block(x, t)
            elif isinstance(block, SelfAttention2d):
                x = rearrange(x, "b c t h w -> (b t) c h w")
                x = block(x)
                x = rearrange(x, "(b t) c h w -> b c t h w", t=num_frames)
            else:
                x = block(x)
        for block in self.up_blocks:
            if isinstance(block, BasicBlock3d):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block(x, t)
            else:
                x = block(x)

        x = x + residual
        for block in self.out_blocks:
            if isinstance(block, BasicBlock3d):
                x = block(x, t)
            else:
                x = block(x)
        return x
