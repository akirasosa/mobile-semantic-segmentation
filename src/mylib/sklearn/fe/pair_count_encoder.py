import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
import numpy as np


# https://www.kaggle.com/matleonard/categorical-encodings
class PairCountEncoder(TransformerMixin):
    def __init__(self, n_components=3, seed=123):
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self.svd_encoding = None

    def fit(self, X, y=None):
        df = pd.concat((
            pd.DataFrame(X.values, columns=['main', 'sub']),
            pd.DataFrame(np.ones(len(X)), columns=['y'])
        ), axis=1)
        pair_counts = df.groupby(['main', 'sub'])['y'].count()
        mat = pair_counts.unstack(fill_value=0)
        self.svd_encoding = pd.DataFrame(self.svd.fit_transform(mat), index=mat.index)
        return self

    def transform(self, X, y=None):
        return self.svd_encoding.reindex(X.values[:, 0]).values
