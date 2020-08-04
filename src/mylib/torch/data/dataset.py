import pandas as pd
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.to_dict()

    def __len__(self):
        return len(self.df)
