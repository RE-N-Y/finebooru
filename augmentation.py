# Original credits to:
# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
from einops import rearrange, reduce

def brightness(x):
    x = x + (torch.rand(len(x), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def saturation(x):
    mean = reduce(x, 'b c h w -> b 1 h w', 'mean')
    x = (x - mean) * (torch.rand(len(x), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean
    return x


def contrast(x):
    mean = reduce(x, 'b c h w -> b 1 1 1', 'mean')
    x = (x - mean) * (torch.rand(len(x), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean
    return x


def translation(x, ratio=0.125):
    b, c, h, w = x.shape

    sx, xy = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[len(x), 1, 1], device=x.device)
    ty = torch.randint(-xy, xy + 1, size=[len(x), 1, 1], device=x.device)
    batch, gx, gy= torch.meshgrid(
        torch.arange(len(x), dtype=torch.long, device=x.device),
        torch.arange(h, dtype=torch.long, device=x.device),
        torch.arange(w, dtype=torch.long, device=x.device),
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy= torch.clamp(gy+ ty + 1, 0, w + 1)
    padding = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])

    x = rearrange(padding, 'b h w c -> b h w c')
    x = rearrange(x[batch, gx, gy], 'b h w c -> b c h w')

    return x.contiguous()


def cutout(x, ratio=0.5):
    b, c, h, w = x.shape

    cuty, cutx = int(h * ratio + 0.5), int(w * ratio + 0.5)
    offx = torch.randint(0, h + (1 - cuty % 2), size=[len(x), 1, 1], device=x.device)
    offy = torch.randint(0, w + (1 - cutx % 2), size=[len(x), 1, 1], device=x.device)
    batch, gx, gy = torch.meshgrid(
        torch.arange(len(x), dtype=torch.long, device=x.device),
        torch.arange(cuty, dtype=torch.long, device=x.device),
        torch.arange(cutx, dtype=torch.long, device=x.device),
    )
    gx = torch.clamp(gx + offx - cuty // 2, min=0, max=h - 1)
    gy = torch.clamp(gy + offy - cutx // 2, min=0, max=w - 1)
    mask = torch.ones(len(x), h, w, dtype=x.dtype, device=x.device)
    mask[batch, gx, gy] = 0
    x = x * rearrange(mask, 'b h w -> b 1 h w')

    return x


AUGMENTATIONS = {
    'color': [brightness, saturation, contrast],
    'translation': [translation],
    'cutout': [cutout],
}

def daugment(x):
    policy='color'
    for p in policy.split(','):
        for f in AUGMENTATIONS[p]:
            x = f(x)
    return x