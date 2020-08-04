from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mobile_seg.const import DATA_LFW_DIR
from mylib.albumentations.augmentations.transforms import MyCoarseDropout


def _mask_to_img(mask_file: Path) -> Path:
    return DATA_LFW_DIR / f'raw/images/{mask_file.stem}.jpg'


def _img_to_mask(img_file: Path) -> Path:
    return DATA_LFW_DIR / f'raw/masks/{img_file.stem}.ppm'


def get_img_files() -> np.ndarray:
    mask_files = sorted(DATA_LFW_DIR.glob('**/*.ppm'))
    return np.array(list(map(_mask_to_img, mask_files)))


class MaskDataset(Dataset):
    def __init__(
            self,
            img_files: np.ndarray,
            transform: A.Compose,
            mask_axis: int = 0,  # 0 is hair
    ):
        self.img_files = img_files
        self.mask_files = [_img_to_mask(f) for f in img_files]
        self.transform = transform
        self.mask_axis = mask_axis

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        img = np.array(img)

        mask = Image.open(self.mask_files[idx])
        mask = np.array(mask)
        mask = mask[:, :, self.mask_axis]

        augmented = self.transform(image=img, mask=mask)

        img = np.array(augmented['image']).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(augmented['mask']).astype(np.float32)

        return img / 255., mask / 255.

    def __len__(self):
        return len(self.img_files)


# %%
if __name__ == '__main__':
    # %%
    import matplotlib.pyplot as plt

    # %%
    img_files = get_img_files()
    img_size = 224
    dataset = MaskDataset(
        img_files,
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
