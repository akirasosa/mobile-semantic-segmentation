import torch
import torch.nn as nn


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x = torch.max(x, x1)
        return x


if __name__ == '__main__':
    frelu = FReLU(16)
    inputs = torch.randn((2, 16, 32, 32))
    print(frelu(inputs).shape)
