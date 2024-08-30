import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as onp
from functools import partial
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from huggingface_hub import PyTorchModelHubMixin

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
    def __init__(self, features, codes, pages, beta:float=0.25, **kwargs):
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
    
class LFQuantiser(nn.Module):
    def __init__(self, features:int, codes:int, pages:int, beta:float=0.25, temperature:float=0.1, **kwargs):
        super().__init__()

        self.codes = int(math.log2(pages))
        self.input = nn.Linear(features, self.codes)
        self.output = nn.Linear(self.codes, features)
        self.beta = beta
        
        mask = torch.exp2(torch.arange(self.codes - 1, -1, -1))
        self.temperature = temperature
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("usage", torch.arange(pages))

    def forward(self, z):
        z = F.normalize(self.input(z), dim=-1)
        ones = torch.ones(z.shape, device=z.device, dtype=z.dtype)
        codes = torch.where(z > 0, ones, -ones) # b t d
        idxes = reduce((codes > 0) * self.mask[None, None, :], 'b t d -> b t', 'sum')
        codes = F.normalize(codes, dim=-1)

        loss = self.beta * torch.mean((z - codes.detach()) ** 2)
        codes = z + (codes - z).detach()
        codes = self.output(codes)

        return codes, loss, idxes
    

class FSQuantiser(nn.Module):
    def __init__(self, features:int, codes:int, pages:int, levels:list[int] = [], beta:float=0.25, **kwargs):
        super().__init__()

        basis = onp.concatenate(([1], onp.cumprod(levels[:-1])))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.int))
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int))

        codes = len(levels)
        self.codes = codes
        self.pages = onp.prod(levels)

        self.input = nn.Linear(features, codes)
        self.output = nn.Linear(codes, features)
        self.eps = 1e-3

    def bound(self, z):
        half = (self.levels - 1) * (1 + self.eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)
        shift = torch.atanh(offset / half)
        return torch.tanh(z + shift) * half - offset
    
    def quantize(self, z:Tensor) -> Tensor:
        z = self.bound(z)
        quantized = z + (torch.round(z) - z).detach()

        # Renormalize to [-1, 1].
        half = self.levels // 2
        return quantized / half
    
    def indexes(self, codes:Tensor) -> Tensor:
        half = self.levels // 2
        codes = codes * half + half
        idxes = torch.sum(codes * self.basis, dim=-1)
        return idxes.to(torch.uint32)
    
    def codes(self, idxes:Tensor) -> Tensor:
        half = self.levels // 2
        idxes = rearrange(idxes, '... -> ... 1')
        codes = (idxes // self.basis) % self.levels
        codes = (codes - half) / codes
        return codes
    
    def forward(self, z):
        z = self.input(z)
        codes = self.quantize(z)
        idxes = self.indexes(codes)
        z = self.output(codes)
        
        return z, 0, idxes

class WPE(nn.Module):
    def __init__(self, features:int, length:int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(length, features))

    def forward(self, x):
        return x + self.embeddings

# RMSNorm implementation taken from LLaMA repository
class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    @torch.compile
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class CRMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, features, 1, 1))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
    
    @torch.compile
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, features:int, useconv:bool=False, ntokens:int=32, bias=False):
        super().__init__()
        self.gate = nn.Linear(features, 4 * features, bias=bias)
        self.up = nn.Linear(features, 4 * features, bias=bias)
        self.down = nn.Linear(4 * features, features, bias=bias)
        self.convolve = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=ntokens, w=ntokens),
            nn.Conv2d(4 * features, 4 * features, 3, padding=1, groups=4 * features),
            Rearrange('b c h w -> b (h w) c')
        ) if useconv else nn.Identity()

    @torch.compile
    def forward(self, x):
        return self.down(self.convolve(F.silu(self.gate(x)) * self.up(x)))
    

class CSwiGLU(nn.Module):
    def __init__(self, features:int, bias=False):
        super().__init__()
        self.gate = nn.Conv2d(features, 4 * features, 1, padding=0, bias=bias)
        self.up = nn.Conv2d(features, 4 * features, 1, padding=0, bias=bias)
        self.down = nn.Conv2d(4 * features, features, 1, padding=0, bias=bias)

    @torch.compile
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Attention(nn.Module):
    def __init__(self, features:int, heads:int, bias=False):
        super().__init__()
        # attention implementation
        self.attention = nn.MultiheadAttention(features, heads, bias=bias, batch_first=True)
        self.out = nn.Linear(features, features)

    def forward(self, x):
        x, attn = self.attention(x, x, x) # q k v
        x = self.out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, features, heads:int=12, bias=False, useconv=False, ntokens:int=32):
        super().__init__()
        self.attention = Attention(features, heads, bias=bias)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.mlp = SwiGLU(features, bias=bias, useconv=useconv, ntokens=ntokens)

    def forward(self, x):
        x = self.attention(self.prenorm(x)) + x
        x = self.mlp(self.postnorm(x)) + x
        return x


class NeXtformer(nn.Module):
    def __init__(self, features:int, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(features, features, 3, padding=1, groups=features, bias=False)
        self.prenorm = CRMSNorm(features)
        self.postnorm = CRMSNorm(features)
        self.mlp = CSwiGLU(features, bias=False)

    @torch.compile
    def forward(self, x):
        x = self.depthwise(self.prenorm(x)) + x
        x = self.mlp(self.postnorm(x)) + x

        return x  

class CVQVAE(
        nn.Module, 
        PyTorchModelHubMixin, 
        library_name="finebooru",
        repo_url="https://github.com/RE-N-Y/finebooru"
    ):
    def __init__(self, features:int=192, heads:int=12, codes:int=32, pages:int=8192, size:int=256, levels=4, bias=False, quantiser="vq", dims:int=[8,8,8,6,5], temperature=0.1):
        super().__init__()
        self.size = size

        mults = [1, 1, 2, 2, 4, 4]
        maxmult = mults[levels - 1]

        self.levels = levels
        self.mults = mults[:levels]
        self.stacks = 3
        self.input = nn.Conv2d(3, features, 3, padding=1, bias=bias)

        layers = []
        for ins, outs in zip(self.mults[:-1], self.mults[1:]):
            layers.extend([Downsample(), nn.Conv2d(features * ins, features * outs, 3, padding=1, bias=bias)])
            layers.extend([NeXtformer(features * outs) for _ in range(self.stacks)])
            size = size // 2

        layers.extend([Rearrange('b c h w -> b (h w) c'), Transformer(features * maxmult, heads=heads, bias=bias)])

        self.encoder = nn.Sequential(*layers)
        if quantiser == "vq":
            self.quantiser = VectorQuantiser(features * maxmult, codes, pages)
        elif quantiser == "fsq":
            self.quantiser = FSQuantiser(features * maxmult, codes, pages, levels=dims)
        else:
            raise ValueError(f"Unknown quantiser {quantiser}")

        layers = []
        layers.extend([Transformer(features * maxmult, heads=heads, bias=bias), Rearrange('b (h w) c -> b c h w', h=size, w=size)])

        rmults = self.mults[::-1]
        for ins, outs in zip(rmults[:-1], rmults[1:]):
            size *= 2
            layers.extend([NeXtformer(features * ins) for _ in range(self.stacks)])
            layers.extend([Upsample(), nn.Conv2d(features * ins, features * outs, 3, padding=1, bias=bias)])
        
        self.decoder = nn.Sequential(*layers)
        self.output = nn.Conv2d(features, 3, 3, padding=1, bias=bias)

    def forward(self, x):
        x = self.encoder(self.input(x))
        codes, loss, idxes = self.quantiser(x)
        x = self.output(self.decoder(codes))

        return x, loss, idxes
    
    def encode(self, x):
        x = self.encode(self.input(x))
        codes, loss, idxes = self.quantiser(x)
        return codes
    
    def tokenise(self, x):
        x = self.encode(self.input(x))
        codes, loss, idxes = self.quantiser(x)
        return idxes
    
    def decode(self, codes):
        return self.output(self.decoder(codes))


class VQVAE(nn.Module):
    def __init__(
        self, 
        features:int=768, 
        backbone="attention", 
        quantiser="vq", 
        codes:int=32, 
        pages:int=8192,
        heads:int=12,
        depth:int=12,
        patch:int=16,
        size:int=256,
        strides:int=16,
        padding:int=0,
        bias=False,
        temperature=0.1,
        dims:int=[8,8,8,6,5],
        useconv=False
    ):
        super().__init__()
        self.features = features
        self.size = size
        self.patch = patch
        self.strides = strides

        self.ntoken = (size // strides)
        self.epe = WPE(features, self.ntoken ** 2)
        self.dpe = WPE(features, self.ntoken ** 2)

        if backbone == "attention":
            Block = Transformer
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        # patchify
        self.input = nn.Conv2d(3, features, patch, stride=strides, padding=padding, bias=bias)
        # encoder
        transformers = [Block(features, heads=heads, bias=bias, useconv=useconv, ntokens=self.ntoken) for _ in range(depth)]
        self.encoder = nn.Sequential(*transformers, RMSNorm(features), nn.Linear(features, 4 * features), nn.Tanh(), nn.Linear(4 * features, features))

        # quantiser
        if quantiser == "vq":
            self.quantiser = VectorQuantiser(features, codes, pages)
        elif quantiser == "fsq":
            self.quantiser = FSQuantiser(features, codes, pages, levels=dims)
        else:
            raise ValueError(f"Unknown quantiser {quantiser}")
        
        # decoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.decoder = nn.Sequential(*transformers, RMSNorm(features), nn.Linear(features, 4 * features), nn.Tanh(), nn.Linear(4 * features, features))
        self.output = nn.Sequential(
            nn.Conv2d(features, 3 * patch ** 2, 1, bias=bias),
            nn.PixelShuffle(patch)
        )

    def forward(self, x):
        x = rearrange(self.input(x), 'b c h w -> b (h w) c')

        x = self.encoder(self.epe(x))
        codes, loss, idxes = self.quantiser(x)
        x = self.decoder(self.dpe(codes))

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.ntoken, w=self.ntoken)
        x = self.output(x)

        return x, loss, idxes


class TikTok(nn.Module):
    def __init__(self, features:int=768, backbone="attention", quantiser="vq", codes:int=32, pages:int=8192, heads:int=12, depth:int=12, patch:int=16, size:int=256, strides:int=16, padding:int=0, bias=False, temperature=0.1, dims:int=[8,5,5,5]):
        super().__init__()
        self.features = features
        self.size = size
        self.patch = patch
        self.strides = strides

        self.ntoken = (size // strides)
        self.tokens = 32
        scale = 1 / math.sqrt(features)
        self.latent = nn.Parameter(scale * torch.randn(self.tokens, features))
        self.mask = nn.Parameter(scale * torch.randn(features))

        self.epe = WPE(features, self.ntoken ** 2 + self.tokens)
        self.dpe = WPE(features, self.ntoken ** 2 + self.tokens)

        if backbone == "attention":
            Block = Transformer
        else:
            raise ValueError(f"Unsupported backbone {backbone}")

        # patchify
        self.input = nn.Conv2d(3, features, patch, stride=strides, padding=padding, bias=bias)
        # encoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.encoder = nn.Sequential(*transformers)
        # quantiser
        if quantiser == "vq":
            self.quantiser = VectorQuantiser(features, codes, pages)
        elif quantiser == "fsq":
            self.quantiser = FSQuantiser(features, codes, pages, levels=dims)
        else:
            raise ValueError(f"Unknown quantiser {quantiser}")

        # decoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.decoder = nn.Sequential(*transformers)

    def forward(self, x, passthrough=False):
        b, c, h, w = x.shape
        x = rearrange(self.input(x), 'b c h w -> b (h w) c')
        latents = repeat(self.latent, 't d -> b t d', b=b)
        x = torch.cat((x, latents), dim=1)
        x = self.encoder(self.epe(x))
        
        # only quantise the latents
        codes, loss, idxes = self.quantiser(x[:, -self.tokens:, :])
        x = x[:, :-self.tokens, :] if passthrough else codes
        masks = repeat(self.mask, 'd -> b t d', b=b, t=self.ntoken ** 2)
        x = torch.cat((masks, x), dim=1)
        x = self.decoder(self.dpe(x))

        # only output the non-latent part
        x = rearrange(x[:, :-self.tokens, :], 'b (h w) c -> b c h w', h=self.ntoken, w=self.ntoken)

        return x, loss, idxes