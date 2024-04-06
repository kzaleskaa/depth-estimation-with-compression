import torch.nn as nn
import torchvision


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = torchvision.models.efficientnet_b0()
        features = efficientnet.features
        self.layer1 = features[:3]
        self.layer2 = features[3]
        self.layer3 = features[4]

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3
