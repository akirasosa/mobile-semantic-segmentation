import copy
from typing import TypeVar

import torch
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)


def create_ema(src_model: T) -> T:
    ema_model = copy.deepcopy(src_model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    msd = model.state_dict()
    for k, ema_v in ema_model.state_dict().items():
        model_v = msd[k].detach()
        ema_v.copy_(ema_v * decay + (1. - decay) * model_v)
