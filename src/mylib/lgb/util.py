from typing import List

import pandas as pd
from lightgbm import Booster


def make_imp_df(boosters: List[Booster]) -> pd.DataFrame:
    df = pd.concat([
        pd.DataFrame({'name': b.feature_name(), 'importance': b.feature_importance()})
        for b in boosters
    ])
    return df.groupby('name').mean() \
        .sort_values('importance') \
        .reset_index(level='name') \
        .set_index('name')
