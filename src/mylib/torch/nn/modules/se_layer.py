import torch
from timm.models.layers import Mish
from torch import nn

from mylib.torch.nn.modules.dense import Dense


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            Dense(channel, channel // reduction, bias=False),
            Mish(inplace=True),
            Dense(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    inputs = torch.randn((3, 12, 768, 1))
    m = SELayer(12)
    # %%
    m(inputs).shape
