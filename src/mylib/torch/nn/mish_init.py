import math

import torch


def init_weights(m, variance=1.0):
    def _calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            return 1, 1

        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def _initialize_weights(tensor, variance, filters=1):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        gain = variance / math.sqrt(fan_in * filters)
        with torch.no_grad():
            torch.nn.init.normal_(tensor)
            return tensor.data * gain

    def _initialize_bias(tensor, variance):
        with torch.no_grad():
            torch.nn.init.normal_(tensor)
            return tensor.data * variance

    if m is None:
        return
    if hasattr(m, 'weight') and m.weight is not None:
        # We want to avoid initializing Batch Normalization
        if hasattr(m, 'running_mean'):
            return

        # If we have channels we probably are a Convolutional Layer
        filters = 1
        if hasattr(m, 'in_channels'):
            filters = m.in_channels

        m.weight.data = _initialize_weights(m.weight, variance=variance, filters=filters)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data = _initialize_bias(m.bias, variance=variance)
