import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mobile_seg.const import EXP_DIR
from mobile_seg.dataset import load_df, MaskDataset, split_df
from mobile_seg.modules.net import load_trained_model
from mobile_seg.params import DataParams


def get_loader(params: DataParams) -> DataLoader:
    df = load_df()
    _, df_val = split_df(df, params)

    dataset = MaskDataset(
        df_val,
        transform=A.Compose([
            A.Resize(
                params.img_size,
                params.img_size,
            ),
        ]),
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )


# %%
if __name__ == '__main__':
    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = EXP_DIR / 'mobile_seg/1607075632/checkpoints/last.ckpt'

    loader = get_loader(DataParams(
        batch_size=1,
        fold=0,
        n_splits=5,
        img_size=224,
        seed=1,
    ))
    model = load_trained_model(ckpt_path).to(device).eval()

    # %%
    with torch.no_grad():
        inputs, labels = next(iter(loader))

        outputs = model(inputs.to(device)).cpu()

        inputs = inputs.squeeze()
        labels = labels.squeeze()
        outputs = outputs.squeeze()

        inputs = (inputs * 255).numpy().transpose((1, 2, 0)).astype(np.uint8)

        plt.subplot(131)
        plt.imshow(inputs)
        plt.subplot(132)
        plt.imshow(labels)
        plt.subplot(133)
        plt.imshow(outputs)
        plt.show()
