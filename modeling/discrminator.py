import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from einops.layers.torch import Rearrange
from .base import Downsample

class Convolution(nn.Module):
    def __init__(self, features:int, filters:int, kernel:int=3, padding:int=1, down=False, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(filters, features, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(filters, 1, 1)) if bias else None
        self.down = Downsample() if down else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)
        self.scale = 1 / math.sqrt(features * kernel ** 2)
        self.padding = padding

    def forward(self, x, gain=1):
        weight = self.weight * self.scale
        x = F.conv2d(x, weight, padding=self.padding)
        x = self.down(x)
        x = x + self.bias if self.bias is not None else x
        x = self.activation(x) * gain

        return x

class Resnet(nn.Module):
    def __init__(self, features:int, filters:int):
        super().__init__()
        self.skip = Convolution(features, filters, kernel=1, padding=0, down=True, bias=False)
        self.input = Convolution(features, features, kernel=3, padding=1)
        self.out = Convolution(features, filters, kernel=3, padding=1, down=True)
        
    def forward(self, x):
        y = self.skip(x)
        x = self.input(x, gain=math.sqrt(2))
        x = self.out(x)
        x = y + x

        return x

class MiniBatchStd(nn.Module):
    def __init__(self, group:int=4, channels:int=1):
        super().__init__()
        self.group = group
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.shape

        y = rearrange(x, '(g n) (f c) h w -> g n f c h w', g = self.group, f = self.channels)
        y = y - reduce(y, 'g ... -> ...', 'mean') # 'g ... -> ...'
        y = reduce(y ** 2, 'g ... -> ...', 'mean') # 'g ... -> ...'
        y = torch.sqrt(y + 1e-8)
        y = reduce(y, 'n f c h w -> n f', 'mean') # 'n f c h w -> n f'
        y = repeat(y, 'n f -> (g n) f h w', h=h, w=w, g=self.group)
        x, _ = pack([x, y], 'n * h w')

        return x

class Discriminator(nn.Module):
    def __init__(self, size:int=256):
        super().__init__()
        self.filters = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }
        self.stem = nn.Conv2d(3, self.filters[size], 3, padding=1)
        
        logs = int(math.log2(size))
        layers = [ Resnet(self.filters[2 ** i], self.filters[2 ** (i - 1)]) for i in range(logs, 2, -1) ]
        self.net = nn.Sequential(*layers)
        self.out = nn.Sequential(*[
            MiniBatchStd(group=4, channels=1),
            Convolution(self.filters[4] + 1, self.filters[4], kernel=3, padding=1),
            Rearrange('b c h w -> b (c h w) 1 1'),
            Convolution(4 * 4 * self.filters[4], self.filters[4], kernel=1, padding=0),
            Convolution(self.filters[4], 1, kernel=1, padding=0),
        ])

    def forward(self, x):
        x = self.stem(x)
        x = self.net(x)
        x = self.out(x)
        x = x.view(len(x),-1)

        return x