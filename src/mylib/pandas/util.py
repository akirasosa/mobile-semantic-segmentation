import numpy as np
import pandas as pd


def cast_64(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == 'float64':
            df[c] = df[c].astype(np.float32)
        if df[c].dtype == 'int64':
            df[c] = df[c].astype(np.int32)
    return df
