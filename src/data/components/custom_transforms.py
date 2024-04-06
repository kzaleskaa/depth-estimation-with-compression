import torch
import torch.nn.functional as F


class NormalizeData:
    def __init__(self, divisor):
        self.divisor = divisor

    def __call__(self, img):
        return img / self.divisor


class BilinearInterpolation:
    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        return F.interpolate(
            torch.unsqueeze(mask, 0), size=self.size, mode="bilinear", align_corners=True
        )
