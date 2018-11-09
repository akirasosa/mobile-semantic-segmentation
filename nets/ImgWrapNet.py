import torch
import torch.nn as nn

from nets.MobileNetV2_unet import MobileNetV2_unet


def _init_unet(state_dict):
    unet = MobileNetV2_unet(pre_trained=None)
    unet.load_state_dict(state_dict)
    return unet


class ImgWrapNet(nn.Module):
    def __init__(self, state_dict, scale=255.):
        super().__init__()
        self.scale = scale
        self.unet = _init_unet(state_dict)

    def forward(self, x):
        x = x / self.scale
        x = self.unet(x)
        x = x * self.scale
        x = torch.cat((x, x, x), dim=1)
        return x


if __name__ == '__main__':
    WEIGHT_PATH = 'outputs/train_unet/0-best.pth'
    net = ImgWrapNet(torch.load(WEIGHT_PATH, map_location='cpu'))
    net(torch.randn(1, 3, 224, 224))
