import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as onp
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        filter = [1., 3., 3., 1.]
        filter = torch.tensor(filter, dtype=torch.float32)
        filter = filter[:, None] * filter[None, :]
        filter /= filter.sum()
        self.register_buffer('filter', filter)

    def forward(self, x):
        b, c, h, w = x.shape
        kernel = repeat(self.filter, 'h w -> c 1 h w', c=c)
        x = F.conv2d(x, kernel.clone(), stride=2, padding=1, groups=c)
        return x

class Upsample(nn.Module):
    def __init__(self, scale:float=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.interpolate(x, (h * self.scale, w * self.scale))
        return x

class GumbelQuantiser(nn.Module):
    def __init__(self, features:int, codes:int, pages:int, temperature:float=1, klw:float=5e-4):
        super().__init__()

        self.input = nn.Linear(features, codes)
        self.codebook = nn.Embedding(pages, codes)
        self.output = nn.Linear(codes, features)
        self.pages = pages
        self.klw = klw
        self.temperature = temperature

    def forward(self, z):
        hard = self.training
        logits = self.input(z)
        idxes = F.gumbel_softmax(logits, tau=self.temperature, hard=hard) @ self.codebook.weight.T

        codes = self.codebook(idxes)
        scores = F.softmax(logits)
        divergence = self.klw * torch.sum(scores * torch.log(scores * self.pages + 1e-12), dim=-1)

        codes = self.output(codes)

        return codes, divergence.mean(), idxes

class VectorQuantiser(nn.Module):
    def __init__(self, features, codes, pages, beta:float=0.25):
        super().__init__()
        self.codes = codes
        self.pages = pages
        self.beta = beta
        self.input = nn.Linear(features, codes, bias=False)
        self.output = nn.Linear(codes, features, bias=False)
        self.codebook = nn.Embedding(pages, codes, _weight=torch.randn(pages, codes))

    def forward(self, z):
        z = self.input(z)
        z, codes = F.normalize(z, dim=-1), F.normalize(self.codebook.weight, dim=-1)
        distance = rearrange(z, 'b n d -> b n () d') - rearrange(codes, 'c d -> () () c d')
        distance = torch.sum(distance ** 2, axis=-1)

        idxes = torch.argmin(distance, dim=-1)
        codes = self.codebook(idxes)
        codes = F.normalize(codes, dim=-1)

        loss = torch.mean((z.detach() - codes) ** 2) + self.beta * torch.mean((z - codes.detach()) ** 2)
        codes = z + (codes - z).detach()
        codes = self.output(codes)

        return codes, loss, idxes

class WPE(nn.Module):
    def __init__(self, features:int, length:int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(length, features))

    def forward(self, x):
        return x + self.embeddings

# RMSNorm implementation taken from LLaMA repository
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, features:int, bias=True):
        super().__init__()
        self.gate = nn.Linear(features, 4 * features)
        self.up = nn.Linear(features, 4 * features)
        self.down = nn.Linear(4 * features, features)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, features:int, heads:int, bias=True):
        super().__init__()
        # attention implementation
        self.attention = nn.MultiheadAttention(features, heads, bias=bias)
        self.out = nn.Linear(features, features)

    def forward(self, x):
        x, attn = self.attention(x, x, x) # q k v
        x = self.out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, features, heads:int=12, bias=True):
        super().__init__()
        self.attention = Attention(features, heads, bias=bias)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.mlp = SwiGLU(features)

    def forward(self, x):
        x = self.attention(self.prenorm(x)) + x
        x = self.mlp(self.postnorm(x)) + x
        return x

class VQVAE(nn.Module):
    def __init__(self, features:int=768, codes:int=32, pages:int=8192, heads:int=12, depth:int=12, patch:int=16, size:int=256, bias=True):
        super().__init__()
        self.size = size
        self.patch = patch

        ntoken = size//patch
        self.epe = WPE(features, ntoken ** 2)
        self.dpe = WPE(features, ntoken ** 2)

        # patchify
        self.input = nn.Conv2d(3, features, kernel_size=patch, stride=patch, bias=bias)
        # encoder
        transformers = [Transformer(features, heads=heads, bias=bias) for _ in range(depth)]
        self.encoder = nn.Sequential(*[*transformers, nn.LayerNorm(features), nn.Linear(features, 4 * features), nn.Tanh(), nn.Linear(4 * features, features)])
        # quantiser
        self.quantiser = VectorQuantiser(features, codes, pages)
        # decoder
        transformers = [Transformer(features, heads=heads, bias=bias) for _ in range(depth)]
        self.decoder = nn.Sequential(*[*transformers, nn.LayerNorm(features), nn.Linear(features, 4 * features), nn.Tanh(), nn.Linear(4 * features, features)])
        # pixelshuffle
        self.output = nn.Conv2d(features, 3 * patch * patch, kernel_size=1, bias=bias)

    def decode(self, idxes):
        ratio = self.size // self.patch
        codes = self.quantiser.codebook(idxes)
        codes = self.quantiser.output(codes)

        x = self.decoder(codes + self.dpe)
        x = rearrange(x, 'b c (h w) -> b c h w', h=ratio, w=ratio)
        x = rearrange(self.output(x), 'b (c hr wr) h w  -> b c (h hr) (w wr)', hr=self.patch, wr=self.patch)

        return x

    def forward(self, x):
        ratio = self.size // self.patch
        x = rearrange(self.input(x), 'b c h w -> b (h w) c')

        x = self.encoder(self.epe(x))
        codes, loss, idxes = self.quantiser(x)
        x = self.decoder(self.dpe(x))

        x = rearrange(x, 'b (h w) c -> b c h w', h=ratio, w=ratio)
        x = rearrange(self.output(x), 'b (c hr wr) h w  -> b c (h hr) (w wr)', hr=self.patch, wr=self.patch)

        return x, loss, idxes