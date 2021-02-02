from functools import partial
from inspect import isfunction

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence

# helpers


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exits(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=4.):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, seq_len, casual=True, heads=8, dim_head=64, dropout=0., noncasual_att_len=0):
        super.__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head * -0.5

        self.casual = casual
        self.noncasual_attn_len = noncasual_att_len

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = torch.finfo(dots.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b i -> b () i ()') * \
                rearrange(mask, 'b j -> b () () j')
            dots = masked_fill_(~mask, mask_value)
            del mask

        if self.casual:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()

            if self.noncasual_attn_len > 1:
                ind = slice(0, self.noncasual_attn_len)
                mask[ind, ind] = False

            dots.maskked_fill_(mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
