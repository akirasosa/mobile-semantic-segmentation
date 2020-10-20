import dataclasses
from functools import partial

import numpy as np


def flat_cos(
        step: int,
        total_steps: int,
        flat_rate: float = 1.,
        cos_rate: float = 0.72,
):
    flat_steps = int(flat_rate / (flat_rate + cos_rate) * total_steps)
    cos_steps = total_steps - flat_steps
    if step <= flat_steps:
        return 1
    f = np.cos((step - flat_steps) / cos_steps * np.pi) * 0.5 + 0.5
    return np.clip(f, 0, 1)


@dataclasses.dataclass
class _Linear:
    total_steps: int
    start: float
    stop: float
    flat_rate_pre: float
    flat_rate_post: float

    def __post_init__(self):
        steps_pre = int(self.flat_rate_pre * total_steps)
        steps_post = int(self.flat_rate_post * total_steps)
        linear_steps = total_steps - steps_pre - steps_post
        self.schedule = np.concatenate((
            np.ones(steps_pre) * self.start,
            np.linspace(self.start, self.stop, linear_steps),
            np.ones(steps_post) * self.stop,
        ))

    def __call__(self, step: int):
        return self.schedule[step]


def linear(total_steps: int, start: float = 0., stop: float = 1., flat_rate_pre: float = 0.,
           flat_rate_post: float = 0.):
    return _Linear(total_steps, start, stop, flat_rate_pre, flat_rate_post)


# %%
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt

    # %%
    total_steps = 100
    # sched = linear(total_steps, flat_rate_pre=0.1, flat_rate_post=0.2)
    # sched = flat_cos(total_steps)
    sched = partial(flat_cos, total_steps=total_steps)
    values = [sched(n) for n in range(total_steps)]

    plt.plot(range(total_steps), values)
    plt.show()
