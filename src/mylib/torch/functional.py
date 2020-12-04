import torch
import torch.nn.functional as F


@torch.jit.script
def calculate_distances(p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
    # ReLU prevents negative numbers in sqrt
    Dij = torch.sqrt(F.relu(torch.sum((p0 - p1) ** 2, -1)))
    return Dij


def calculate_torsions(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    if p0.dim() == 1:
        b1 /= b1.norm()
    else:
        b1 /= b1.norm(dim=1)[:, None]

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v) * w, dim=-1)

    return torch.atan2(y, x)


# %%
if __name__ == '__main__':
    # %%
    coords = torch.tensor([[10.396, 18.691, 19.127],
                           [9.902, 18.231, 20.266],
                           [8.736, 17.274, 20.226],
                           [7.471, 18.048, 19.846]])
    coords2 = torch.tensor([[7.471, 18.048, 19.846],
                            [6.67, 17.583, 18.852],
                            [5.494, 18.412, 18.503],
                            [4.59, 18.735, 19.711]])

    print(calculate_torsions(*coords))
    print(calculate_torsions(*coords2))
    # %%
    # calculate_torsions(*coords[:, None, :])
    a = torch.cat((coords, coords2), 1).reshape(4, -1, 3)
    print(calculate_torsions(*a))
