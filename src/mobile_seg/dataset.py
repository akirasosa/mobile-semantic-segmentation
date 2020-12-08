from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from mobile_seg.const import DATA_LFW_DIR, CACHE_DIR
from mobile_seg.params import DataParams
from mylib.albumentations.augmentations.transforms import MyCoarseDropout
from mylib.pandas.cache import pd_cache


def _mask_to_img(mask_file: Path) -> Path:
    return DATA_LFW_DIR / f'raw/images/{mask_file.stem}.jpg'


@pd_cache(CACHE_DIR, ext='.pqt')
def load_df():
    mask_files = list(sorted(DATA_LFW_DIR.glob('**/*.ppm')))
    img_files = list(map(_mask_to_img, mask_files))
    return pd.DataFrame({
        'img_path': map(str, img_files),
        'mask_path': map(str, mask_files),
    })


def split_df(df: pd.DataFrame, params: DataParams):
    folds = KFold(
        n_splits=params.n_splits,
        random_state=params.seed,
        shuffle=True,
    )
    train_idx, val_idx = list(folds.split(df))[params.fold]
    return df.iloc[train_idx], df.iloc[val_idx]


class MaskDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            transform: A.Compose,
            mask_axis: int = 0,  # 0 is hair
    ):
        self.df = df
        self.transform = transform
        self.mask_axis = mask_axis

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row['img_path'])
        img = np.array(img)

        mask = Image.open(row['mask_path'])
        mask = np.array(mask)
        mask = mask[:, :, self.mask_axis]

        augmented = self.transform(image=img, mask=mask)

        img = np.array(augmented['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(augmented['mask']).astype(np.float32)

        return img / 255., mask / 255.

    def __len__(self):
        return len(self.df)


# %%
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt

    # %%
    df = load_df()
    img_size = 224
    dataset = MaskDataset(
        df,
        transform=A.Compose([
            A.RandomResizedCrop(
                img_size,
                img_size,
            ),
            A.Rotate(13),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
            A.RGBShift(),
            A.RandomGamma(),
            # A.CLAHE(),
            MyCoarseDropout(
                min_holes=1,
                max_holes=8,
                max_height=32,
                max_width=32,
            ),
            # A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            # ToTensorV2(),
        ]),
    )
    img, mask = dataset[0]
    print(img.shape, img.mean(), img.std())
    print(mask.shape, mask.min(), mask.max())

    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.imshow((img * 255).transpose((1, 2, 0)).astype(np.uint8))
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(mask.squeeze())
    plt.show()
