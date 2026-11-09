import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.Softplus(beta=100), nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))

    def forward(self, x):
        return x + self.res(x)

class DRUNet_light(nn.Module):

    def __init__(self, in_nc=1):
        super().__init__()
        out_nc = in_nc
        nc = [64, 128, 256, 512]
        nb = 2
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, padding=1, bias=False)
        self.m_down1 = nn.Sequential(*[ResBlock(nc[0], nc[0]) for _ in range(nb)], nn.Conv2d(nc[0], nc[1], 2, stride=2, bias=False))
        self.m_down2 = nn.Sequential(*[ResBlock(nc[1], nc[1]) for _ in range(nb)], nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, bias=False))
        self.m_down3 = nn.Sequential(*[ResBlock(nc[2], nc[2]) for _ in range(nb)], nn.Conv2d(nc[2], nc[3], 2, stride=2, bias=False))
        self.m_body = nn.Sequential(*[ResBlock(nc[3], nc[3]) for _ in range(nb)])
        self.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, stride=2, bias=False), *[ResBlock(nc[2], nc[2]) for _ in range(nb)])
        self.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, stride=2, bias=False), *[ResBlock(nc[1], nc[1]) for _ in range(nb)])
        self.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, stride=2, bias=False), *[ResBlock(nc[0], nc[0]) for _ in range(nb)])
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, padding=1, bias=False)

    def forward(self, x):
        (_, _, h, w) = x.size()
        (r1, r2) = (h % 8, w % 8)
        x = F.pad(x, pad=(0, 8 - r2 if r2 > 0 else 0, 0, 8 - r1 if r1 > 0 else 0), mode='reflect')
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        out = self.m_body(x4)
        out = self.m_up3(out + x4)
        out = self.m_up2(out + x3)
        out = self.m_up1(out + x2)
        out = self.m_tail(out + x1)
        return out[..., :h, :w]
