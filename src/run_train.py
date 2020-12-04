from functools import cached_property, partial
from logging import getLogger, FileHandler
from multiprocessing import cpu_count
from os import cpu_count
from pathlib import Path
from time import time
from typing import Optional, Union, Sequence, Dict

import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch_optimizer import RAdam

from mobile_seg.dataset import get_img_files, MaskDataset
from mobile_seg.loss import dice_loss
from mobile_seg.modules.net import MobileNetV2_unet
from mobile_seg.params import ModuleParams, Params, DataParams
from mylib.albumentations.augmentations.transforms import MyCoarseDropout
from mylib.pytorch_lightning.base_module import PLBaseModule, StepResult
from mylib.pytorch_lightning.logging import configure_logging
from mylib.torch.ensemble.ema import create_ema
from mylib.torch.optim.sched import flat_cos


# noinspection PyAbstractClass
class DataModule(pl.LightningDataModule):
    def __init__(self, params: DataParams):
        super().__init__()
        self.params = params
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        img_files = get_img_files()

        folds = KFold(
            n_splits=self.params.n_splits,
            random_state=self.params.seed,
            shuffle=True,
        )
        train_idx, val_idx = list(folds.split(img_files))[self.params.fold]

        self.train_dataset = MaskDataset(
            img_files[train_idx],
            transform=A.Compose([
                A.RandomResizedCrop(
                    self.params.img_size,
                    self.params.img_size,
                ),
                A.Rotate(13),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RGBShift(),
                A.RandomGamma(),
                MyCoarseDropout(
                    min_holes=1,
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                ),
            ]),
        )
        self.val_dataset = MaskDataset(
            img_files[val_idx],
            transform=A.Compose([
                A.Resize(
                    self.params.img_size,
                    self.params.img_size,
                ),
            ]),
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, Sequence[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
        )


# noinspection PyAbstractClass
class PLModule(PLBaseModule[MobileNetV2_unet]):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.hparams = hparams
        self.model = MobileNetV2_unet(
            drop_rate=self.hp.drop_rate,
            drop_path_rate=self.hp.drop_path_rate,
        )
        self.criterion = dice_loss(scale=2)
        if self.hp.use_ema:
            self.ema_model = create_ema(self.model)

    def step(self, model: MobileNetV2_unet, batch) -> StepResult:
        X, y = batch
        y_hat = model.forward(X)
        # assert y.shape == y_hat.shape, f'{y.shape}, {y_hat.shape}'

        loss = self.criterion(y_hat, y)
        n_processed = len(y)

        return {
            'loss': loss,
            'n_processed': n_processed,
        }

    def configure_optimizers(self):
        params = self.model.parameters()

        if self.hp.optim == 'adam':
            opt = Adam(
                params,
                lr=self.hp.lr,
                weight_decay=self.hp.weight_decay,
            )
            sched = {
                'scheduler': OneCycleLR(
                    opt,
                    max_lr=self.hp.lr,
                    total_steps=self.total_steps,
                ),
                'interval': 'step',
            }
        elif self.hp.optim == 'radam':
            opt = RAdam(
                params,
                lr=self.hp.lr,
                weight_decay=self.hp.weight_decay,
            )
            # noinspection PyTypeChecker
            sched = {
                'scheduler': LambdaLR(
                    opt,
                    lr_lambda=partial(
                        flat_cos,
                        total_steps=self.total_steps,
                    ),
                ),
                'interval': 'step',
            }
        else:
            raise Exception

        return [opt], [sched]

    @cached_property
    def hp(self) -> ModuleParams:
        return ModuleParams.from_dict(dict(self.hparams))


def train(params: Params):
    seed_everything(params.d.seed)

    tb_logger = TensorBoardLogger(
        params.t.save_dir,
        name=f'mobile_seg',
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger('lightning')
    logger.addHandler(FileHandler(log_dir / 'train.log'))
    logger.info(params.pretty())

    trainer = pl.Trainer(
        max_epochs=params.t.epochs,
        gpus=params.t.gpus,
        tpu_cores=params.t.num_tpu_cores,
        logger=tb_logger,
        precision=params.t.precision,
        resume_from_checkpoint=params.t.resume_from_checkpoint,
        weights_save_path=params.t.save_dir,
        callbacks=[
            # EarlyStopping(
            #     monitor='ema_0_loss' if params.m.use_ema else 'val_0_loss',
            #     patience=3,
            #     mode='min'
            # ),
            ModelCheckpoint(
                monitor='ema_0_loss' if params.m.use_ema else 'val_0_loss',
                save_last=True,
                verbose=True,
            ),
        ],
        checkpoint_callback=True,
        deterministic=True,
        benchmark=True,
    )
    net = PLModule(params.m.to_dict())
    dm = DataModule(params.d)

    trainer.fit(net, datamodule=dm)


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    if params.do_cv:
        for p in params.copy_for_cv():
            train(p)
    else:
        train(params)
