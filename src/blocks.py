import math

from einops import parse_shape, rearrange
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


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length=10000):
        super().__init__()
        self.register_buffer("embedding", self.make_embedding(dim, max_length))

    def forward(self, x):
        # Parameters
        #   x: (bsz,) discrete
        return self.embedding[x]

    @staticmethod
    def make_embedding(dim, max_length=10000):
        embedding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding


class FourierEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("freqs", self.make_freqs(dim, max_period))

    def forward(self, x):
        # Parameters
        #   x: (bsz,) continuous
        outer = torch.outer(x, self.freqs)
        embedding = torch.cat([torch.cos(outer), torch.sin(outer)], dim=-1)
        return embedding

    @staticmethod
    def make_freqs(dim, max_period=10000):
        return torch.exp(torch.arange(0, dim, 2) * -math.log(max_period) / dim)


class FFN(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.init_embed = nn.Linear(in_dim, embed_dim)
        self.time_embed = PositionalEmbedding(embed_dim)
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
                BasicBlock(in_c, in_c, embed_dim),
                BasicBlock(in_c, in_c, embed_dim),
                nn.Conv2d(in_c, out_c, 3, 2, 1) if not is_last else nn.Conv2d(in_c, out_c, 1),
            ]))

        for idx, (in_c, out_c, skip_c) in enumerate(zip(
            all_dims[::-1][:-1],
            all_dims[::-1][1:],
            all_dims[:-1][::-1],
        )):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(nn.ModuleList([
                BasicBlock(in_c + skip_c, in_c, embed_dim),
                BasicBlock(in_c + skip_c, in_c, embed_dim),
                nn.ConvTranspose2d(in_c, out_c, 2, 2) if not is_last else nn.Conv2d(in_c, out_c, 1),
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


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention [Vaswani et al. NeurIPS 2017].
    Scaled dot-product attention is performed over V, using K as keys and Q as queries.
        MultiHeadAttention(Q, V) = FC(SoftMax(1/√d QKᵀ) V) (concatenated over multiple heads),
    Notes
    -----
    (1) Q, K, V can be of different dimensions. Q and K are projected to dim_a and V to dim_o.
    (2) We assume the last and second last dimensions correspond to the feature (i.e. embedding)
        and token (i.e. words) dimensions respectively.
    """
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout_prob=0.1, dim_a=None, dim_o=None):
        super().__init__()
        if dim_a is None:
            dim_a = dim_q
        if dim_o is None:
            dim_o = dim_q
        self.dim_a, self.dim_o, self.num_heads = dim_a, dim_o, num_heads
        self.fc_q = nn.Linear(dim_q, dim_a, bias=True)
        self.fc_k = nn.Linear(dim_k, dim_a, bias=True)
        self.fc_v = nn.Linear(dim_v, dim_o, bias=True)
        self.fc_o = nn.Linear(dim_o, dim_o, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc_q, self.fc_k, self.fc_v, self.fc_o):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, q, k, v, mask=None):
        """
        Perform multi-head attention with given queries and values.
        Parameters
        ----------
        q: (bsz, tsz, dim_q)
        k: (bsz, tsz, dim_k)
        v: (bsz, tsz, dim_v)
        mask: (bsz, tsz) or (bsz, tsz, tsz), where 1 denotes keep and 0 denotes remove
        Returns
        -------
        O: (bsz, tsz, dim_o)
        """
        bsz, tsz, _ = q.shape
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        q = torch.cat(q.split(self.dim_a // self.num_heads, dim=-1), dim=0)
        k = torch.cat(k.split(self.dim_a // self.num_heads, dim=-1), dim=0)
        v = torch.cat(v.split(self.dim_o // self.num_heads, dim=-1), dim=0)
        a = q @ k.transpose(-1, -2) / self.dim_a ** 0.5
        if mask is not None:
            assert mask.ndim in (2, 3)
            if mask.ndim == 3:
                mask = mask.repeat(self.num_heads, 1, 1)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-2).repeat(self.num_heads, tsz, 1)
            a.masked_fill_(mask == 0, -65504)
        a = self.dropout(torch.softmax(a, dim=-1))
        o = self.fc_o(torch.cat((a @ v).split(bsz, dim=0), dim=-1))
        return o


class PositionwiseFFN(nn.Module):
    """
    Position-wise FFN [Vaswani et al. NeurIPS 2017].
    """
    def __init__(self, dim, hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        for module in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderBlock(nn.Module):
    """
    Transformer encoder block [Vaswani et al. NeurIPS 2017].
    Note that this is the pre-LN version [Nguyen and Salazar 2019].
    """
    def __init__(self, dim, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, dim, dim, num_heads, dropout_prob)
        self.ffn = PositionwiseFFN(dim, hidden_dim, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp_time_1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.mlp_time_2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, t, mask=None):
        x_ = self.ln1(x)
        x_ = x_ + self.mlp_time_1(t).unsqueeze(1)
        x = x + self.dropout(self.attn(x_, x_, x_, mask))
        x_ = self.ln2(x)
        x_ = x_ + self.mlp_time_2(t).unsqueeze(1)
        x = x + self.dropout(self.ffn(x_))
        return x


class ViT(nn.Module):
    """
    Simple version of Vision Transformer [Dosovitsky et al. 2020].
    """
    def __init__(self, in_dim, embed_dim, num_layers, patch_shape):
        super().__init__()

        self.patch_h, self.patch_w = patch_shape
        self.patch_embed = nn.Linear(in_dim * math.prod(patch_shape), embed_dim, bias=True)
        self.time_embed = nn.Parameter(PositionalEmbedding.make_embedding(embed_dim))
        self.pos_embed = nn.Parameter(PositionalEmbedding.make_embedding(embed_dim))

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, 4 * embed_dim) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.out_embed = nn.Linear(embed_dim, in_dim * math.prod(patch_shape), bias=True)

    def forward(self, x, t):

        shape_info = parse_shape(x, "b c h w")

        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=self.patch_h, pw=self.patch_w)
        x = self.patch_embed(x) + rearrange(self.pos_encoding[:x.shape[1]], "t c -> () t c")
        t = self.time_embed[t]

        for block in self.blocks:
            x = block(x, t)

        x = self.ln(x)
        x = self.out_embed(x)

        x = rearrange(x, "b (h w) (ph pw c) -> b c (h ph) (w pw)",
            h=shape_info["h"] // self.patch_h,
            w=shape_info["w"] // self.patch_w,
            ph=self.patch_h,
            pw=self.patch_w,
        )
        return x
