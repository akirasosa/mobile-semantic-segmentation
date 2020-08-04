import numpy as np
import pandas as pd
from nancorrmp.nancorrmp import NaNCorrMp


def calc_corr(df: pd.DataFrame) -> pd.DataFrame:
    corr = NaNCorrMp.calculate(df.select_dtypes('number'))
    corr = corr.abs().unstack().sort_values(ascending=False).reset_index()
    corr.columns = ['col1', 'col2', 'val']
    df_corr = corr[corr.col1 > corr.col2]
    return df_corr.reset_index()


def find_high_corr(df_corr: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    df_tmp = df_corr[(df_corr.col1 == col) | (df_corr.col2 == col)].head(n).copy()
    df_tmp.loc[df_corr.col1 == col, 'col1'] = np.nan
    df_tmp.loc[df_corr.col2 == col, 'col2'] = np.nan
    df_tmp['col'] = df_tmp.col1.combine_first(df_tmp.col2)
    return df_tmp[['col', 'val']]
