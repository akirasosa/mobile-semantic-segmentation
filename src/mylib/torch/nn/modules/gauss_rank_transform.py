import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class GaussRankTransform(nn.Module):
    def __init__(self, data: torch.Tensor, eps=1e-6):
        super(GaussRankTransform, self).__init__()
        tformed = self._erfinv(data, eps)
        data, sort_idx = data.sort()
        self.register_buffer('src', data)
        self.register_buffer('dst', tformed[sort_idx])

    @staticmethod
    def _erfinv(data: torch.Tensor, eps):
        rank = data.argsort().argsort().float()

        rank_scaled = (rank / rank.max() - 0.5) * 2
        rank_scaled = rank_scaled.clamp(-1 + eps, 1 - eps)

        tformed = rank_scaled.erfinv()

        return tformed

    def forward(self, x):
        return self._transform(x, self.dst, self.src)

    def invert(self, x):
        return self._transform(x, self.src, self.dst)

    def _transform(self, x, src, dst):
        pos = src.argsort()[x.argsort().argsort()]

        N = len(self.src)
        pos[pos >= N] = N - 1
        pos[pos - 1 <= 0] = 0

        x1 = dst[pos]
        x2 = dst[pos - 1]
        y1 = src[pos]
        y2 = src[pos - 1]

        relative = (x - x2) / (x1 - x2)

        return (1 - relative) * y2 + relative * y1


# %%
if __name__ == '__main__':
    # %%
    x = torch.from_numpy(np.random.uniform(low=0, high=1, size=2000))

    grt = GaussRankTransform(x)
    x_tformed = grt.forward(x)
    x_inv = grt.invert(x_tformed)

    # %%
    print(x)
    print(x_inv)

    print(grt.dst)
    print(torch.sort(x_tformed)[0])

    bins = 100
    plt.hist(x, bins=bins)
    plt.show()

    plt.hist(x_inv, bins=bins)
    plt.show()

    plt.hist(grt.src, bins=bins)
    plt.show()

    plt.hist(x_tformed, bins=bins)
    plt.show()

    plt.hist(grt.dst, bins=bins)
    plt.show()
