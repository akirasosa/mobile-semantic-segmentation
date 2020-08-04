from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer


class KBinsStratifiedKFold(StratifiedKFold):
    def __init__(
            self,
            n_splits=5,
            *,
            shuffle=False,
            random_state=None,
            n_bins: int = 5,
            strategy: str = 'quantile',
    ):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
        self.kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        super()._iter_test_indices()

    def split(self, X, y, groups=None):
        y_binned = self.kbd.fit_transform(y)
        return super().split(X, y_binned, groups)


# %%
if __name__ == '__main__':
    # %%
    import numpy as np

    skf = KBinsStratifiedKFold(n_splits=5, shuffle=True, random_state=123, n_bins=10)
    y = np.random.random(100).reshape(-1, 1)
    # %%
    list(skf.split(y, y))
