import numbers
from typing import List, Optional, Iterable, Union

import category_encoders as ce
import numpy as np
import pandas as pd
from category_encoders.utils import convert_input, convert_input_vector
from sklearn import model_selection
from sklearn.base import BaseEstimator, clone, TransformerMixin
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target


def check_cv(cv: Union[int, Iterable, BaseCrossValidator] = 5,
             y: Optional[Union[pd.Series, np.ndarray]] = None,
             stratified: bool = False,
             random_state: int = 0):
    if cv is None:
        cv = 5
    if isinstance(cv, numbers.Integral):
        if stratified and (y is not None) and (type_of_target(y) in ('binary', 'multiclass')):
            return StratifiedKFold(cv, shuffle=True, random_state=random_state)
        else:
            return KFold(cv, shuffle=True, random_state=random_state)

    return model_selection.check_cv(cv, y, stratified)


class KFoldEncoderWrapper(BaseEstimator, TransformerMixin):
    """KFold Wrapper for sklearn like interface

    This class wraps sklearn's TransformerMixIn (object that has fit/transform/fit_transform methods),
    and call it as K-fold manner.

    Args:
        base_transformer:
            Transformer object to be wrapped.
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.

            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        return_same_type:
            If True, `transform` and `fit_transform` return the same type as X.
            If False, these APIs always return a numpy array, similar to sklearn's API.
    """

    def __init__(self, base_transformer: BaseEstimator,
                 cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None, return_same_type: bool = True,
                 groups: Optional[pd.Series] = None):
        self.cv = cv
        self.base_transformer = base_transformer

        self.n_splits = None
        self.transformers = None
        self.return_same_type = return_same_type
        self.groups = groups

    def _pre_train(self, y):
        self.cv = check_cv(self.cv, y)
        self.n_splits = self.cv.get_n_splits()
        self.transformers = [clone(self.base_transformer) for _ in range(self.n_splits + 1)]

    def _fit_train(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params) -> pd.DataFrame:
        if y is None:
            X_ = self.transformers[-1].transform(X)
            return self._post_transform(X_)

        X_ = X.copy()

        for i, (train_index, test_index) in enumerate(self.cv.split(X_, y, self.groups)):
            self.transformers[i].fit(X.iloc[train_index], y.iloc[train_index], **fit_params)
            X_.iloc[test_index, :] = self.transformers[i].transform(X.iloc[test_index])
        self.transformers[-1].fit(X, y, **fit_params)

        return X_

    def _post_fit(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return X

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit models for each fold.

        Args:
            X:
                Data
            y:
                Target
        Returns:
            returns the transformer object.
        """
        self._post_fit(self.fit_transform(X, y), y)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform X

        Args:
            X: Data

        Returns:
            Transformed version of X. It will be pd.DataFrame If X is `pd.DataFrame` and return_same_type is True.
        """
        is_pandas = isinstance(X, pd.DataFrame)
        X_ = self._fit_train(X, None)
        X_ = self._post_transform(X_)
        return X_ if self.return_same_type and is_pandas else X_.values

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: pd.Series = None, **fit_params) \
            -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit models for each fold, then transform X

        Args:
            X:
                Data
            y:
                Target
            fit_params:
                Additional parameters passed to models

        Returns:
            Transformed version of X. It will be pd.DataFrame If X is `pd.DataFrame` and return_same_type is True.
        """
        assert len(X) == len(y)
        self._pre_train(y)

        is_pandas = isinstance(X, pd.DataFrame)
        X = convert_input(X)
        y = convert_input_vector(y, X.index)

        if y.isnull().sum() > 0:
            # y == null is regarded as test data
            X_ = X.copy()
            X_.loc[~y.isnull(), :] = self._fit_train(X[~y.isnull()], y[~y.isnull()], **fit_params)
            X_.loc[y.isnull(), :] = self._fit_train(X[y.isnull()], None, **fit_params)
        else:
            X_ = self._fit_train(X, y, **fit_params)

        X_ = self._post_transform(self._post_fit(X_, y))

        return X_ if self.return_same_type and is_pandas else X_.values


class TargetEncoder(KFoldEncoderWrapper):
    """Target Encoder

    KFold version of category_encoders.TargetEncoder in
    https://contrib.scikit-learn.org/categorical-encoding/targetencoder.html.

    Args:
        cv:
            int, cross-validation generator or an iterable which determines the cross-validation splitting strategy.

            - None, to use the default ``KFold(5, random_state=0, shuffle=True)``,
            - integer, to specify the number of folds in a ``(Stratified)KFold``,
            - CV splitter (the instance of ``BaseCrossValidator``),
            - An iterable yielding (train, test) splits as arrays of indices.
        groups:
            Group labels for the samples. Only used in conjunction with a “Group” cv instance (e.g., ``GroupKFold``).
        cols:
            A list of columns to encode, if None, all string columns will be encoded.
        drop_invariant:
            Boolean for whether or not to drop columns with 0 variance.
        handle_missing:
            Options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        handle_unknown:
            Options are ‘error’, ‘return_nan’ and ‘value’, defaults to ‘value’, which returns the target mean.
        min_samples_leaf:
            Minimum samples to take category average into account.
        smoothing:
            Smoothing effect to balance categorical average vs prior. Higher value means stronger regularization.
            The value must be strictly bigger than 0.
        return_same_type:
            If True, ``transform`` and ``fit_transform`` return the same type as X.
            If False, these APIs always return a numpy array, similar to sklearn's API.
    """

    def __init__(self, cv: Optional[Union[Iterable, BaseCrossValidator]] = None,
                 groups: Optional[pd.Series] = None,
                 cols: List[str] = None,
                 drop_invariant: bool = False, handle_missing: str = 'value', handle_unknown: str = 'value',
                 min_samples_leaf: int = 1, smoothing: float = 1.0, return_same_type: bool = True):
        e = ce.TargetEncoder(cols=cols, drop_invariant=drop_invariant, return_df=True,
                             handle_missing=handle_missing,
                             handle_unknown=handle_unknown,
                             min_samples_leaf=min_samples_leaf, smoothing=smoothing)

        super().__init__(e, cv, return_same_type, groups)

    def _post_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self.transformers[0].cols
        for c in cols:
            X[c] = X[c].astype(float)
        return X
