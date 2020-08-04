import torch
import torch.nn as nn


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)
