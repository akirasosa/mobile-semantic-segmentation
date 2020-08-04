from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.efficientnet import mobilenetv2_100
from timm.models.efficientnet_builder import efficientnet_init_weights


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UpSampleBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super(UpSampleBlock, self).__init__()
        self.dconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.invres = InvertedResidual(out_channels * 2, out_channels, 1, 6)

    def forward(self, x0, x1):
        x = torch.cat([
            x0,
            self.dconv(x1)
        ], dim=1)
        x = self.invres(x)
        return x


class MobileNetV2_unet(nn.Module):
    def __init__(self, **kwargs):
        super(MobileNetV2_unet, self).__init__()
        self.backbone = mobilenetv2_100(pretrained=True, **kwargs)
        self.up_sample_blocks = nn.ModuleList([
            UpSampleBlock(1280, 96),
            UpSampleBlock(96, 32),
            UpSampleBlock(32, 24),
            UpSampleBlock(24, 16),
        ])
        self.conv_last = nn.Sequential(
            nn.Conv2d(16, 3, 1),
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid(),
        )
        del self.backbone.bn2, self.backbone.act2, self.backbone.global_pool, self.backbone.classifier

        efficientnet_init_weights(self.up_sample_blocks)
        efficientnet_init_weights(self.conv_last)

    def forward(self, x):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)

        down_feats = []
        for b in self.backbone.blocks:
            x = b(x)
            if x.shape[1] in [16, 24, 32, 96]:
                down_feats.append(x)
        x = self.backbone.conv_head(x)

        for (f, b) in zip(reversed(down_feats), self.up_sample_blocks):
            x = b(f, x)

        x = self.conv_last(x)

        return x


def load_trained_model(ckpt_path: Path) -> MobileNetV2_unet:
    state_dict: OrderedDict = torch.load(ckpt_path)['state_dict']
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        new_dict[k[6:]] = v

    model = MobileNetV2_unet()
    model.load_state_dict(new_dict)
    return model


# %%
if __name__ == '__main__':
    # %%
    model = MobileNetV2_unet()
    inputs = torch.randn((2, 3, 224, 224))
    out = model(inputs)

    print(model)
    print(out.shape)
