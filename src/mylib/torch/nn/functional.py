from typing import Optional, Tuple

import torch
from torch_scatter import scatter_sum
from torch_scatter.utils import broadcast


@torch.jit.script
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count_ret = count.clone()
    count.clamp_(1)
    count = broadcast(count, out, dim)
    out.div_(count)
    return out, count_ret


@torch.jit.script
def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim: int = 0, descending: bool = False, eps: float = 1e-12):
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm,


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output
