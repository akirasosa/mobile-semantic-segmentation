import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomResizedCrop, RandomRotation, RandomHorizontalFlip, ToTensor, \
    Resize, RandomAffine, ColorJitter

from dataset import MaskDataset, get_img_files
from loss import dice_loss
from nets.MobileNetV2_unet import MobileNetV2_unet
from trainer import Trainer

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

# %%
N_CV = 5
BATCH_SIZE = 8
LR = 1e-4

N_EPOCHS = 100
IMG_SIZE = 224
RANDOM_STATE = 1

EXPERIMENT = 'train_unet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)


# %%
def get_data_loaders(train_files, val_files, img_size=224):
    train_transform = Compose([
        ColorJitter(0.3, 0.3, 0.3, 0.3),
        RandomResizedCrop(img_size, scale=(0.8, 1.2)),
        RandomAffine(10.),
        RandomRotation(13.),
        RandomHorizontalFlip(),
        ToTensor(),
    ])
    # train_mask_transform = Compose([
    #     RandomResizedCrop(img_size, scale=(0.8, 1.2)),
    #     RandomAffine(10.),
    #     RandomRotation(13.),
    #     RandomHorizontalFlip(),
    #     ToTensor(),
    # ])
    val_transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
    ])

    train_loader = DataLoader(MaskDataset(train_files, train_transform),
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)
    val_loader = DataLoader(MaskDataset(val_files, val_transform),
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    return train_loader, val_loader


def save_best_model(cv, model, df_hist):
    if df_hist['val_loss'].tail(1).iloc[0] <= df_hist['val_loss'].min():
        torch.save(model.state_dict(), '{}/{}-best.pth'.format(OUT_DIR, cv))


def write_on_board(writer, df_hist):
    row = df_hist.tail(1).iloc[0]

    writer.add_scalars('{}/loss'.format(EXPERIMENT), {
        'train': row.train_loss,
        'val': row.val_loss,
    }, row.epoch)


def log_hist(df_hist):
    last = df_hist.tail(1)
    best = df_hist.sort_values('val_loss').head(1)
    summary = pd.concat((last, best)).reset_index(drop=True)
    summary['name'] = ['Last', 'Best']
    logger.debug(summary[['name', 'epoch', 'train_loss', 'val_loss']])
    logger.debug('')


def run_cv(img_size, pre_trained):
    image_files = get_img_files()
    kf = KFold(n_splits=N_CV, random_state=RANDOM_STATE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        train_files = image_files[train_idx]
        val_files = image_files[val_idx]

        writer = SummaryWriter()

        def on_after_epoch(m, df_hist):
            save_best_model(n, m, df_hist)
            write_on_board(writer, df_hist)
            log_hist(df_hist)

        criterion = dice_loss(scale=2)
        data_loaders = get_data_loaders(train_files, val_files, img_size)
        trainer = Trainer(data_loaders, criterion, device, on_after_epoch)

        model = MobileNetV2_unet(pre_trained=pre_trained)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=LR)

        hist = trainer.train(model, optimizer, num_epochs=N_EPOCHS)
        hist.to_csv('{}/{}-hist.csv'.format(OUT_DIR, n), index=False)

        writer.close()

        break


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(logging.FileHandler(filename="outputs/{}.log".format(EXPERIMENT)))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='image size',
    )
    parser.add_argument(
        '--pre_trained',
        type=str,
        help='path of pre trained weight',
    )
    args, _ = parser.parse_known_args()
    print(args)
    # run_cv(**vars(args))

