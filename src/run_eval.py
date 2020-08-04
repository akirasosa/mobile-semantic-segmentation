import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from mobile_seg.const import EXP_DIR
from mobile_seg.dataset import MaskDataset, get_img_files
from mobile_seg.modules.net import load_trained_model
from mobile_seg.params import ModuleParams, Params


def get_loader(params: ModuleParams) -> DataLoader:
    img_files = get_img_files()

    folds = KFold(
        n_splits=params.n_splits,
        random_state=params.seed,
        shuffle=True,
    )
    _, val_idx = list(folds.split(img_files))[params.fold]

    val_dataset = MaskDataset(
        img_files[val_idx],
        transform=A.Compose([
            A.Resize(
                params.img_size,
                params.img_size,
            ),
        ]),
    )

    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
    )


# %%
if __name__ == '__main__':
    # %%
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = EXP_DIR / 'mobile_seg/1596704750/checkpoints/epoch=194.ckpt'
    params = Params.load('params/001.yaml')

    loader = get_loader(params.module_params)
    model = load_trained_model(ckpt_path).to(device).eval()

    # %%
    with torch.no_grad():
        inputs, labels = next(iter(loader))

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        inputs = inputs.squeeze()
        inputs = (inputs * 255).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

        labels = labels.squeeze().cpu()
        outputs = outputs.squeeze().cpu()

        plt.subplot(131)
        plt.imshow(inputs)
        plt.subplot(132)
        plt.imshow(labels)
        plt.subplot(133)
        plt.imshow(outputs)
        plt.show()
