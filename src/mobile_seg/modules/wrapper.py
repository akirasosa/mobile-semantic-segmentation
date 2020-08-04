import torch
import torch.nn as nn

from mobile_seg.modules.net import MobileNetV2_unet


class Wrapper(nn.Module):
    def __init__(
            self,
            unet: MobileNetV2_unet,
            scale: float = 255.
    ):
        super().__init__()
        self.unet = unet
        self.scale = scale

    def forward(self, x):
        x = x / self.scale
        x = self.unet(x)
        x = x * self.scale
        x = torch.cat((x, x, x), dim=1)
        return x


# %%
if __name__ == '__main__':
    # %%
    model = MobileNetV2_unet()
    wrapper = Wrapper(model)
    inputs = torch.randn((1, 3, 224, 224))
    out = wrapper(inputs)
    print(out.shape)
