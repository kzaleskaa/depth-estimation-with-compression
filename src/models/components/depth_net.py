import torch
import torch.nn as nn

from .bifpn_decoder import BiFPN
from .efficientnet_encoder import EfficientNet


class DepthNet(nn.Module):
    def __init__(self, in_features = [24, 40, 80]):
        super().__init__()
        self.encoder = EfficientNet()
        self.decoder = BiFPN(in_features)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample_2   = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3, padding="same")

    def upsample_cat(self, x):
        p4, p5, p6 = x
        p6 = self.upsample_4(p6)
        p5 = self.upsample_2(p5)
        return torch.cat([p4, p5, p6], dim=1)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        out = self.decoder([x1, x2, x3])
        cated = self.upsample_cat(out)
        res = self.final_conv(cated)
        return res
