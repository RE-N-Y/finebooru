import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as onp
from functools import partial
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba2
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

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
        half = (self.levels - 1) * (1 - self.eps) / 2
        offset = torch.where(self.levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half)
        return torch.tanh(z + shift) * half - offset
    
    def quantize(self, z:Tensor) -> Tensor:
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = z + (torch.round(self.bound(z)) - z).detach()

        # Renormalize to [-1, 1].
        half = self.levels // 2
        return quantized / half
    
    def indexes(self, codes:Tensor) -> Tensor:
        half = self.levels // 2
        codes = codes * half + half
        idxes = torch.sum(codes * self.basis, dim=-1)
        return idxes.to(torch.uint32)
    
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
    def __init__(self, features:int, bias=False):
        super().__init__()
        self.gate = nn.Linear(features, 4 * features, bias=bias)
        self.up = nn.Linear(features, 4 * features, bias=bias)
        self.down = nn.Linear(4 * features, features, bias=bias)

    @torch.compile
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
    

class CSwiGLU(nn.Module):
    def __init__(self, features:int, bias=False):
        super().__init__()
        self.gate = nn.Conv2d(features, 4 * features, 1, padding=0, bias=bias)
        self.up = nn.Conv2d(features, 4 * features, 1, padding=0, bias=bias)
        self.down = nn.Conv2d(4 * features, features, 1, padding=0, bias=bias)

    @torch.compile
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SSD(nn.Module):
    def __init__(self, features:int, heads:int, bias=False):
        super().__init__()
        self.mamba = Mamba2(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.mlp = SwiGLU(features)

    def forward(self, x):
        x = self.mamba(self.prenorm(x)) + x
        x = self.mlp(self.postnorm(x)) + x

        return x

# simple bidirectional mamba similar to Bidirectional LSTM
class BSSD(nn.Module):
    def __init__(self, features:int, heads:int, bias=False):
        super().__init__()

        self.fwd = Mamba2(features)
        self.bwd = Mamba2(features)

        self.prenorm = RMSNorm(features)
        self.fwdnorm = RMSNorm(features)
        self.bwdnorm = RMSNorm(features)

        self.fwdmlp = SwiGLU(features)
        self.bwdmlp = SwiGLU(features)

    def forward(self, x):
        # b t d
        f = x
        b = torch.flip(x, dims=[1])

        f = self.fwd(self.prenorm(f)) + f
        f = self.fwdmlp(self.fwdnorm(f)) + f
        b = self.bwd(self.prenorm(b)) + b
        b = self.bwdmlp(self.bwdnorm(b)) + b

        b = torch.flip(b, dims=[1])

        return f + b

class SSDV(nn.Module):
    def __init__(
        self,
        features,
        heads=8,
        states=64,
        dconv=4,
        expand=2,
        ngroups=1,
        chunk=256,
        bias=False,
    ):
        super().__init__()
        self.features = features
        self.states = states
        self.dconv = dconv
        self.expand = expand
        self.inner = self.expand * self.features
        self.ngroups = ngroups

        self.headdim = (self.inner // 2) // heads
        self.nheads = heads
        self.chunk = chunk

        assert self.inner % self.headdim == 0

        # Order: [z, x, B, C, dt]
        self.inputs = nn.Linear(
            self.features, self.inner + 2 * self.ngroups * self.states + self.nheads, 
            bias=False
        )

        self.xconv = nn.Conv1d(
            self.inner//2 + 2 * self.ngroups * self.states,
            self.inner//2 + 2 * self.ngroups * self.states,
            bias=True,
            kernel_size=dconv,
            groups=self.inner//2 + 2 * self.ngroups * self.states,
            padding=dconv - 1,
        )

        self.zconv = nn.Conv1d(
            self.inner//2,
            self.inner//2,
            bias=True,
            kernel_size=dconv,
            groups=self.inner//2,
            padding=dconv - 1,
        )

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, ) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(1,16)
        Alog = torch.log(A)

        self.Alog = nn.Parameter(Alog)
        self.Alog._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        self.norm = RMSNorm(self.inner, eps=1e-5)
        self.output = nn.Linear(self.inner, self.features, bias=False)

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, _, _ = u.shape

        zxbcdt = self.inputs(u)  # (B, L, D)
        A = -torch.exp(self.Alog)  # (nheads) or (inner, states)

        z, xBC, dt = torch.split(
            zxbcdt, [self.inner // 2, self.inner // 2 + 2 * self.ngroups * self.states, self.nheads], dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2),
            rearrange(self.xconv.weight, "d 1 w -> d w"),
            bias=self.xconv.bias,
            activation="silu"
        ).transpose(1, 2)

        z = causal_conv1d_fn(
            z.transpose(1, 2),
            rearrange(self.zconv.weight, "d 1 w -> d w"),
            bias=self.zconv.bias,
            activation="silu"
        ).transpose(1, 2)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.inner//2, self.ngroups * self.states, self.ngroups * self.states], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk,
            D=self.D,
            z=None,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        y = self.norm(torch.cat([y, z], dim=-1))
        out = self.output(y)
        
        return out
class VSSD(nn.Module):
    def __init__(
        self,
        features,
        heads=8,
        states=64,
        dconv=4,
        expand=2,
        ngroups=1,
        chunk=256,
        bias=False,
    ):
        super().__init__()
        self.features = features
        self.states = states
        self.dconv = dconv
        self.expand = expand
        self.inner = self.expand * self.features
        self.ngroups = ngroups

        self.headdim = self.inner // heads
        self.nheads = heads
        self.chunk = chunk

        assert self.inner % self.headdim == 0

        # Order: [z, x, B, C, dt]
        self.inputs = nn.Linear(
            self.features, 2 * self.inner + 2 * self.ngroups * self.states + self.nheads, 
            bias=False
        )

        self.xconv = nn.Conv1d(
            self.inner + 2 * self.ngroups * self.states,
            self.inner + 2 * self.ngroups * self.states,
            bias=True,
            kernel_size=dconv,
            groups=self.inner + 2 * self.ngroups * self.states,
            padding=dconv - 1,
        )

        self.zconv = nn.Conv1d(
            self.inner,
            self.inner,
            bias=True,
            kernel_size=dconv,
            groups=self.inner,
            padding=dconv - 1,
        )

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, ) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        # A parameter
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(1,16)
        Alog = torch.log(A)

        self.Alog = nn.Parameter(Alog)
        self.Alog._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        self.norm = RMSNorm(self.inner, eps=1e-5)
        self.output = nn.Linear(self.inner, self.features, bias=False)

    def forward(self, u):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, _, _ = u.shape

        zxbcdt = self.inputs(u)  # (B, L, inner)
        A = -torch.exp(self.Alog)  # (nheads) or (inner, states)

        z, xBC, dt = torch.split(
            zxbcdt, [self.inner, self.inner + 2 * self.ngroups * self.states, self.nheads], dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        xBC = causal_conv1d_fn(
            xBC.transpose(1, 2),
            rearrange(self.xconv.weight, "d 1 w -> d w"),
            bias=self.xconv.bias,
            activation="silu"
        ).transpose(1, 2)

        z = causal_conv1d_fn(
            z.transpose(1, 2),
            rearrange(self.zconv.weight, "d 1 w -> d w"),
            bias=self.zconv.bias,
            activation="silu"
        ).transpose(1, 2)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.inner, self.ngroups * self.states, self.ngroups * self.states], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk,
            D=self.D,
            z=None,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y * z)
        out = self.output(y)
        
        return out

class Attention(nn.Module):
    def __init__(self, features:int, heads:int, bias=False):
        super().__init__()
        # attention implementation
        self.attention = nn.MultiheadAttention(features, heads, bias=bias)
        self.out = nn.Linear(features, features)

    def forward(self, x):
        x, attn = self.attention(x, x, x) # q k v
        x = self.out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, features, heads:int=12, bias=False):
        super().__init__()
        self.attention = Attention(features, heads, bias=bias)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.mlp = SwiGLU(features, bias=bias)

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

class CVQVAE(nn.Module):
    def __init__(self, features:int=192, heads:int=12, codes:int=32, pages:int=8192, size:int=256, levels=3, bias=False, quantiser="vq", dims:int=[8,8,8,6,5], temperature=0.1):
        super().__init__()
        self.size = size

        mults = [1, 1, 2, 2, 4]
        maxmult = mults[levels]

        self.levels = levels
        self.mults = mults[:levels]
        self.stacks = 2
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
        elif quantiser == "lfq":
            self.quantiser = LFQuantiser(features * maxmult, codes, pages, temperature=temperature)
        elif quantiser == "gumbel":
            self.quantiser = GumbelQuantiser(features * maxmult, codes, pages)
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


class VQVAE(nn.Module):
    def __init__(self, features:int=768, backbone="attention", quantiser="vq", codes:int=32, pages:int=8192, heads:int=12, depth:int=12, patch:int=16, size:int=256, strides:int=16, padding:int=0, bias=False, temperature=0.1, dims:int=[8,8,8,6,5]):
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
        elif backbone == "ssd":
            Block = SSD
        elif backbone == "bssd":
            Block = BSSD
        elif backbone == "vssd":
            Block = VSSD
        elif backbone == "ssdv":
            Block = SSDV
        else:
            raise ValueError(f"Unknown backbone {backbone}")

        # patchify
        self.input = nn.Conv2d(3, features, patch, stride=strides, padding=padding, bias=bias)
        # encoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.encoder = nn.Sequential(*[*transformers, nn.Linear(features, 4 * features), nn.Tanh(), nn.Linear(4 * features, features)])

        # quantiser
        if quantiser == "vq":
            self.quantiser = VectorQuantiser(features, codes, pages)
        elif quantiser == "lfq":
            self.quantiser = LFQuantiser(features, codes, pages, temperature=temperature)
        elif quantiser == "gumbel":
            self.quantiser = GumbelQuantiser(features, codes, pages)
        elif quantiser == "fsq":
            self.quantiser = FSQuantiser(features, codes, pages, levels=dims)
        else:
            raise ValueError(f"Unknown quantiser {quantiser}")
        
        # decoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.decoder = nn.Sequential(*[*transformers, nn.Linear(features, 4 * features), nn.Tanh(), nn.Linear(4 * features, features)])
        # pixelshuffle
        self.output = nn.Linear(features, 3 * patch * patch, bias=bias)

    def initialise(self):
        def base(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.apply(base)
        nn.init.xavier_uniform_(self.input.weight.view(self.features, -1))


    def forward(self, x):
        x = rearrange(self.input(x), 'b c h w -> b (h w) c')

        x = self.encoder(self.epe(x))
        codes, loss, idxes = self.quantiser(x)
        x = self.decoder(self.dpe(codes))

        x = self.output(x)
        x = rearrange(x, 'b (h w) (c hr wr) -> b c (h hr) (w wr)', h=self.ntoken, w=self.ntoken, hr=self.patch, wr=self.patch, c=3)

        return x, loss, idxes
    
class VARTokenizer(nn.Module):
    def __init__(self, features:int=768, backbone="attention", codes:int=32, pages:int=8192, heads:int=12, depth:int=12, patch:int=16, size:int=256, strides:int=16, padding:int=0, bias=False):
        super().__init__()
        self.size = size
        self.patch = patch

        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for formula
        self.ntoken = (size + 2 * padding - patch) // strides + 1
        self.epe = WPE(features, self.ntoken ** 2)
        self.dpe = WPE(features, self.ntoken ** 2)

        if backbone == "attention":
            Block = Transformer
        elif backbone == "ssd":
            Block = SSD
        elif backbone == "bssd":
            Block = BSSD
        else:
            raise ValueError(f"Unknown backbone {backbone}")
        
        self.scales = [2 ** i for i in range(int(math.log2(size)) )] + [size]

        # patchify
        self.input = nn.Conv2d(3, features, patch, stride=strides, padding=padding, bias=bias)
        # encoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.encoder = nn.Sequential(*[*transformers])
        # quantiser
        self.quantiser = VectorQuantiser(features, codes, pages)

        self.phis = nn.ModuleList([nn.Conv2d(features, features, 3, padding=1) for _ in self.scales])

        # decoder
        transformers = [Block(features, heads=heads, bias=bias) for _ in range(depth)]
        self.decoder = nn.Sequential(*[*transformers])
        # pixelshuffle
        self.output = nn.Conv2d(features, 3 * patch * patch, 1, bias=bias)


    def forward(self, x):
        x = rearrange(self.input(x), 'b c h w -> b (h w) c')
        x = self.encoder(self.epe(x))

        r = losses = 0
        tokens = []

        for scale, phi in zip(self.scales, self.phis):
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.ntoken, w=self.ntoken)
            x = F.interpolate(x, size=(scale, scale), mode="bilinear")
            codes, loss, idxes = self.quantiser
            z = F.interpolate(codes, size=(self.size, self.size), mode="bilinear")
            z = phi(z)

            r, x = r + z, x - z
            losses += loss
            tokens.append(idxes)
        
        r = self.decoder(self.dpe(r))