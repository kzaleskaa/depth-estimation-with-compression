from typing import List

import torch
import torch.nn as nn

from .ffnet_decoder import UpHeadA
from .efficientnet_encoder import EfficientNet


class DepthNet(nn.Module):
    def __init__(self, in_features: List[int] = [24, 40, 80]):
        super().__init__()
        self.encoder = EfficientNet()
        self.decoder = UpHeadA(in_features)
        self.final_conv = nn.Conv2d(in_channels=384, out_channels=1, kernel_size=3, padding="same")

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        x = self.decoder([x1, x2, x3])
        x = self.final_conv(x)
        return x