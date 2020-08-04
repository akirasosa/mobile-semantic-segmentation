from multiprocessing import cpu_count
from typing import Optional, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class NullImpSelection:
    def __init__(self, df: pd.DataFrame, params: Optional[Dict] = None, n_repeat: int = 80):
        self.df = df
        params = params if params is not None else {}
        self.params = {
            'objective': 'regression',
            'boosting_type': 'rf',
            'subsample': 0.623,
            'colsample_bytree': 0.7,
            'num_leaves': 127,
            'max_depth': 8,
            'seed': 123,
            'bagging_freq': 1,
            'n_jobs': cpu_count(),
            'verbose': -1,
            **params,
        }
        self.n_repeat = n_repeat
        self.real_imp_df: Optional[pd.DataFrame] = None
        self.null_imp_df: Optional[pd.DataFrame] = None

    def prepare_imp(self):
        self._make_real_imp()
        self._make_null_imp()

    def _make_real_imp(self):
        self.real_imp_df = self._get_feature_importances(shuffle=False)

    def _make_null_imp(self):
        null_imp_df = pd.DataFrame()
        for i in tqdm(range(self.n_repeat)):
            imp_df = self._get_feature_importances(shuffle=True)
            imp_df['run'] = i + 1
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        self.null_imp_df = null_imp_df

    def get_feature_scores(self) -> pd.DataFrame:
        assert self.real_imp_df is not None, 'Run prepare_imp at first.'

        feature_scores = []
        real_imp = self.real_imp_df
        null_imp = self.null_imp_df

        for _f in real_imp['feature'].unique():
            f_null_imps_gain = null_imp.loc[null_imp['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = real_imp.loc[real_imp['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(
                1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid divide by zero
            f_null_imps_split = null_imp.loc[null_imp['feature'] == _f, 'importance_split'].values
            f_act_imps_split = real_imp.loc[real_imp['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(
                1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid divide by zero
            feature_scores.append((_f, split_score, gain_score))

        return pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

    def get_correlation_scores(self) -> pd.DataFrame:
        assert self.real_imp_df is not None, 'Run prepare_imp at first.'

        correlation_scores = []
        real_imp = self.real_imp_df
        null_imp = self.null_imp_df

        for _f in real_imp['feature'].unique():
            f_null_imps = null_imp.loc[null_imp['feature'] == _f, 'importance_gain'].values
            f_act_imps = real_imp.loc[real_imp['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp.loc[null_imp['feature'] == _f, 'importance_split'].values
            f_act_imps = real_imp.loc[real_imp['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        return pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

    def _get_feature_importances(self, shuffle: bool) -> pd.DataFrame:
        X = self.df.drop(['target'], axis=1)
        y = self.df['target'].copy()
        if shuffle:
            y = self.df['target'].copy().sample(frac=1.0)

        dtrain = lgb.Dataset(X, label=y)
        model = lgb.train(
            params=self.params,
            train_set=dtrain,
            num_boost_round=400,
        )

        x_cols = X.columns

        imp_df = pd.DataFrame()
        imp_df["feature"] = list(x_cols)
        imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
        imp_df["importance_split"] = model.feature_importance(importance_type='split')

        return imp_df
