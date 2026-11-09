import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SkewSymmetricLinear(nn.Linear):

    def __init__(self, in_features, device=None, dtype=None):
        super().__init__(in_features, in_features, bias=False, device=device, dtype=dtype)

    def make_skew_symmetric(self, A):
        return (A - A.T) / 2

    def forward(self, x):
        weight = self.make_skew_symmetric(self.weight)
        return F.linear(x, weight)

class BlockDiagonalMatrixMultiplication(nn.Module):

    def __init__(self, linear, s):
        super().__init__()
        self.linear = linear
        self.s = s

    def forward(self, x, shift, vertical):
        if vertical:
            x = torch.rot90(x, k=1, dims=(-2, -1))
        x = x.roll(shifts=-shift, dims=-1)
        x = rearrange(x, 'n c (h s1) (w s2) -> n h w (c s1 s2)', s1=self.s, s2=self.s)
        x = self.linear(x)
        x = rearrange(x, 'n h w (c s1 s2)   -> n c (h s1) (w s2)', s1=self.s, s2=self.s)
        x = x.roll(shifts=shift, dims=-1)
        if vertical:
            x = torch.rot90(x, k=-1, dims=(-2, -1))
        return x

class CENSURE(nn.Module):

    def __init__(self, model, color=False, n=4):
        super().__init__()
        self.model = model
        self.n = n
        num_features = 16 if not color else 16 * 3
        self.skewsymmetric = BlockDiagonalMatrixMultiplication(SkewSymmetricLinear(num_features), 4)
        self.linear = BlockDiagonalMatrixMultiplication(nn.Linear(num_features, num_features, bias=False), 4)

    def forward(self, x):
        (_, _, h, w) = x.size()
        divisor = 8
        (r1, r2) = (h % divisor, w % divisor)
        x = F.pad(x, pad=(0, divisor - r2 if r2 > 0 else 0, 0, divisor - r1 if r1 > 0 else 0), mode='reflect')
        output = torch.zeros_like(x)
        with torch.enable_grad():
            x.requires_grad_(True)
            out = self.model(x)
            for i in range(self.n):
                for v in [True, False]:
                    y = self.linear(x, i, v)
                    grad = torch.autograd.grad(torch.sum(y ** 2) - torch.sum((y - out) ** 2), x, create_graph=True)[0] / 2
                    output += self.skewsymmetric(grad, i, v)
        return output[..., :h, :w]
