import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

BN_MOMENTUM = 0.1
gpu_up_kwargs = {"mode": "bilinear", "align_corners": True}
mobile_up_kwargs = {"mode": "nearest"}
relu_inplace = True


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        activation=nn.ReLU,
        *args,
        **kwargs,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=ks,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_chan, momentum=BN_MOMENTUM),
        ]
        if activation:
            layers.append(activation(inplace=relu_inplace))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AdapterConv(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[64, 128, 256]):
        super().__init__()
        assert len(in_channels) == len(
            out_channels
        ), "Number of input and output branches should match"
        self.adapter_conv = nn.ModuleList()

        for k in range(len(in_channels)):
            self.adapter_conv.append(
                ConvBNReLU(in_channels[k], out_channels[k], ks=1, stride=1, padding=0),
            )

    def forward(self, x):
        out = []
        for k in range(len(self.adapter_conv)):
            out.append(self.adapter_conv[k](x[k]))
        return out


class UpsampleCat(nn.Module):
    def __init__(self, upsample_kwargs=gpu_up_kwargs):
        super().__init__()
        self._up_kwargs = upsample_kwargs

    def forward(self, x):
        """Upsample and concatenate feature maps."""
        assert isinstance(x, list) or isinstance(x, tuple)
        # print(self._up_kwargs)
        x0 = x[0]
        _, _, H, W = x0.size()
        for i in range(1, len(x)):
            x0 = torch.cat([x0, F.interpolate(x[i], (H, W), **self._up_kwargs)], dim=1)
        return x0


class UpBranch(nn.Module):
    def __init__(
        self,
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_kwargs=gpu_up_kwargs,
    ):
        super().__init__()

        self._up_kwargs = upsample_kwargs

        self.fam_32_sm = ConvBNReLU(in_channels[2], out_channels[2], ks=3, stride=1, padding=1)
        self.fam_32_up = ConvBNReLU(in_channels[2], in_channels[1], ks=1, stride=1, padding=0)
        self.fam_16_sm = ConvBNReLU(in_channels[1], out_channels[0], ks=3, stride=1, padding=1)
        self.fam_16_up = ConvBNReLU(in_channels[1], in_channels[0], ks=1, stride=1, padding=0)
        self.fam_8_sm = ConvBNReLU(in_channels[0], out_channels[0], ks=3, stride=1, padding=1)

        self.high_level_ch = sum(out_channels)
        self.out_channels = out_channels

    def forward(self, x):

        feat8, feat16, feat32 = x

        smfeat_32 = self.fam_32_sm(feat32)
        upfeat_32 = self.fam_32_up(feat32)

        _, _, H, W = feat16.size()
        x = F.interpolate(upfeat_32, (H, W), **self._up_kwargs) + feat16
        smfeat_16 = self.fam_16_sm(x)
        upfeat_16 = self.fam_16_up(x)

        _, _, H, W = feat8.size()
        x = F.interpolate(upfeat_16, (H, W), **self._up_kwargs) + feat8
        smfeat_8 = self.fam_8_sm(x)

        return smfeat_8, smfeat_16, smfeat_32


class UpHeadA(nn.Module):
    def __init__(
        self,
        in_chans,
        base_chans=[64, 128, 256],
        upsample_kwargs=gpu_up_kwargs,
    ):
        layers = []
        super().__init__()
        layers.append(AdapterConv(in_chans, base_chans))
        in_chans = base_chans[:]
        layers.append(UpBranch(in_chans))
        layers.append(UpsampleCat(upsample_kwargs))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
