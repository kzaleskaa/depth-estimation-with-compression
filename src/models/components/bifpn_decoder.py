from typing import List

import torch
import torch.nn as nn


class BiFPN(nn.Module):
    def __init__(self, fpn_sizes: List[int]) -> None:
        super().__init__()

        P4_channels, P5_channels, P6_channels = fpn_sizes
        self.W_bifpn = 64

        self.p6_td_conv = nn.Conv2d(
            P6_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1
        )
        self.p6_td_conv_2 = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p6_td_act = nn.ReLU()
        self.p6_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p6_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

        self.p5_td_conv = nn.Conv2d(
            P5_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1
        )
        self.p5_td_conv_2 = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p5_td_act = nn.ReLU()
        self.p5_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

        self.p4_td_conv = nn.Conv2d(
            P4_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1
        )
        self.p4_td_conv_2 = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p4_td_act = nn.ReLU()
        self.p4_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p3_out_conv_2 = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p3_out_act = nn.ReLU()
        self.p3_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_downsample = nn.MaxPool2d(kernel_size=2)

        self.p4_out_conv = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p4_out_act = nn.ReLU()
        self.p4_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p5_out_conv = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p5_out_act = nn.ReLU()
        self.p5_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_downsample = nn.MaxPool2d(kernel_size=2)
        self.p5_downsample = nn.MaxPool2d(kernel_size=2)

        self.p6_out_conv = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p6_out_act = nn.ReLU()
        self.p6_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p6_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w3 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.p7_out_conv_2 = nn.Conv2d(
            self.W_bifpn,
            self.W_bifpn,
            kernel_size=3,
            stride=1,
            groups=self.W_bifpn,
            bias=True,
            padding=1,
        )
        self.p7_out_act = nn.ReLU()
        self.p7_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p7_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        epsilon = 0.0001
        P4, P5, P6 = inputs

        P6_td_inp = self.p6_td_conv(P6)
        P6_td = self.p6_td_conv_2((self.p6_td_w1 * P6_td_inp) / (self.p6_td_w1 + epsilon))
        P6_td = self.p6_td_act(P6_td)
        P6_td = self.p6_td_conv_bn(P6_td)

        P5_td_inp = self.p5_td_conv(P5)
        P5_td = self.p5_td_conv_2(
            (self.p5_td_w1 * P5_td_inp + self.p5_td_w2 * self.p6_upsample(P6_td))
            / (self.p5_td_w1 + self.p5_td_w2 + epsilon)
        )
        P5_td = self.p5_td_act(P5_td)
        P5_td = self.p5_td_conv_bn(P5_td)

        P4_td_inp = self.p4_td_conv(P4)
        P4_td = self.p4_td_conv_2(
            (self.p4_td_w1 * P4_td_inp + self.p4_td_w2 * self.p5_upsample(P5_td))
            / (self.p4_td_w1 + self.p4_td_w2 + epsilon)
        )
        P4_td = self.p4_td_act(P4_td)
        P4_td = self.p4_td_conv_bn(P4_td)

        P4_out = self.p4_out_conv(
            (self.p4_out_w1 * P4_td_inp + self.p4_out_w2 * P4_td)
            / (self.p4_out_w1 + self.p4_out_w2 + epsilon)
        )
        P4_out = self.p4_out_act(P4_out)
        P4_out = self.p4_out_conv_bn(P4_out)

        P5_out = self.p5_out_conv(
            (
                self.p5_out_w1 * P5_td_inp
                + self.p5_out_w2 * P5_td
                + self.p5_out_w3 * self.p4_downsample(P4_out)
            )
            / (self.p5_out_w2 + self.p5_out_w3 + epsilon)
        )
        P5_out = self.p5_out_act(P5_out)
        P5_out = self.p5_out_conv_bn(P5_out)

        P6_out = self.p6_out_conv(
            (
                self.p6_out_w1 * P6_td_inp
                + self.p6_out_w2 * P6_td
                + self.p6_out_w3 * self.p5_downsample(P5_out)
            )
            / (self.p6_out_w1 + self.p6_out_w2 + self.p6_out_w3 + epsilon)
        )
        P6_out = self.p6_out_act(P6_out)
        P6_out = self.p6_out_conv_bn(P6_out)

        return [P4_out, P5_out, P6_out]
