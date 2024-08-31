from mamba_ssm import Mamba2
from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

class SSD(nn.Module):
    def __init__(self, features:int, heads:int, bias=False, useconv=False, ntokens:int=32):
        super().__init__()
        self.mamba = Mamba2(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.mlp = SwiGLU(features, useconv=useconv, ntokens=ntokens)

    def forward(self, x):
        x = self.mamba(self.prenorm(x)) + x
        x = self.mlp(self.postnorm(x)) + x

        return x

# simple bidirectional mamba similar to Bidirectional LSTM
class BSSD(nn.Module):
    def __init__(self, features:int, heads:int, bias=False, useconv=False, ntokens:int=32):
        super().__init__()

        self.fwd = Mamba2(features)
        self.bwd = Mamba2(features)

        self.prenorm = RMSNorm(features)
        self.fwdnorm = RMSNorm(features)
        self.bwdnorm = RMSNorm(features)

        self.fwdmlp = SwiGLU(features, useconv=useconv, ntokens=ntokens)
        self.bwdmlp = SwiGLU(features, useconv=useconv, ntokens=ntokens)

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