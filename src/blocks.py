import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


def unsqueeze_to(tensor, target_ndim):
    assert tensor.ndim <= target_ndim
    while tensor.ndim < target_ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor

def unsqueeze_as(tensor, target_tensor):
    assert tensor.ndim <= target_tensor.ndim
    while tensor.ndim < target_tensor.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_length=10000):
        super().__init__()
        encoding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return self.encoding[x]


class FFN(nn.Module):

    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.init_embed = nn.Linear(in_dim, embed_dim)
        self.time_embed = PositionalEncoding(embed_dim)
        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, in_dim),
        )

    def forward(self, x, t):
        x = self.init_embed(x)
        t = self.time_embed(t)
        return self.model(x + t)


class BasicBlock(nn.Module):
    """
    BasicBlock: two 3x3 convs followed by a residual connection then ReLU.
    [He et al. CVPR 2016]

        BasicBlock(x) = ReLU( x + Conv3x3( ReLU( Conv3x3(x) ) ) )

    This version supports an additive shift parameterized by time.
    """
    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_c, time_c),
            nn.ReLU(),
            nn.Linear(time_c, out_c),
        )
        if in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x, t):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + unsqueeze_as(self.mlp_time(t), x))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + self.shortcut(x))
        return out


class SelfAttention2d(nn.Module):
    """
    Only implements the MultiHeadAttention component, not the PositionwiseFFN component.
    """
    def __init__(self, dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.o_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = rearrange(q, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        k = rearrange(k, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        v = rearrange(v, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        a = torch.einsum("b c s, b c t -> b s t", q, k) / self.dim ** 0.5
        a = self.dropout(torch.softmax(a, dim=-1))
        o = torch.einsum("b s t, b c t -> b c s", a, v)
        o = rearrange(o, "(b g) c (h w) -> b (g c) h w", g=self.num_heads, w=x.shape[-1])
        return x + self.o_conv(o)


class UNet(nn.Module):
    """
    Simple implementation that closely mimics the one by Phil Wang (lucidrains).
    """
    def __init__(self, in_dim, embed_dim, dim_scales):
        super().__init__()

        self.init_embed = nn.Conv2d(in_dim, embed_dim, 1)
        self.time_embed = PositionalEncoding(embed_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Example:
        #   in_dim=1, embed_dim=32, dim_scales=(1, 2, 4, 8) => all_dims=(32, 32, 64, 128, 256)
        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        for idx, (in_c, out_c) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = idx == len(all_dims) - 2
            self.down_blocks.extend(nn.ModuleList([
                BasicBlock(in_c, in_c, embed_dim),
                BasicBlock(in_c, in_c, embed_dim),
                nn.Conv2d(in_c, out_c, 3, 2, 1) if not is_last else nn.Conv2d(in_c, out_c, 1),
            ]))

        for idx, (in_c, out_c, skip_c) in enumerate(zip(all_dims[::-1][:-1], all_dims[::-1][1:], all_dims[:-1][::-1])):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(nn.ModuleList([
                BasicBlock(in_c + skip_c, in_c, embed_dim),
                BasicBlock(in_c + skip_c, in_c, embed_dim),
                nn.ConvTranspose2d(in_c, out_c, (2, 2), 2) if not is_last else nn.Conv2d(in_c, out_c, 1),
            ]))

        self.mid_blocks = nn.ModuleList([
            BasicBlock(all_dims[-1], all_dims[-1], embed_dim),
            SelfAttention2d(all_dims[-1]),
            BasicBlock(all_dims[-1], all_dims[-1], embed_dim),
        ])
        self.out_blocks = nn.ModuleList([
            BasicBlock(embed_dim, embed_dim, embed_dim),
            nn.Conv2d(embed_dim, in_dim, 1, bias=True),
        ])

    def forward(self, x, t):
        x = self.init_embed(x)
        t = self.time_embed(t)
        skip_conns = []
        residual = x.clone()

        for block in self.down_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
                skip_conns.append(x)
            else:
                x = block(x)
        for block in self.mid_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
            else:
                x = block(x)
        for block in self.up_blocks:
            if isinstance(block, BasicBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block(x, t)
            else:
                x = block(x)

        x = x + residual
        for block in self.out_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
            else:
                x = block(x)
        return x
