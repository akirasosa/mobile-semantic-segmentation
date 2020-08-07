import dataclasses
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, List, Dict, Optional, TypeVar, Type

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_ranger import Ranger
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam

from mylib.torch.optim.sched import flat_cos
from mylib.torch.ensemble.ema import update_ema

T = TypeVar('T')


@dataclasses.dataclass(frozen=True)
class ModuleBaseParams:
    optim: str = 'radam'

    lr: float = 3e-4
    weight_decay: float = 0.

    ema_decay: Optional[float] = None
    ema_eval_freq: int = 1


class PLBaseModule(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.ema_model: Optional[nn.Module] = None
        self.best: float = float('inf')

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, second_order_closure)
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.hp.ema_decay)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        result = self.step(batch, prefix='train')
        return {
            'loss': result['train_loss'],
            **result,
        }

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, prefix='val')

        if self.eval_ema:
            result_ema = self.step(batch, prefix='ema', model=self.ema_model)
        else:
            result_ema = {}

        return {
            **result,
            **result_ema,
        }

    @abstractmethod
    def step(self, batch, prefix: str, model=None) -> Dict:
        pass

    def training_epoch_end(self, outputs):
        metrics = self.collect_metrics(outputs, 'train')
        self.__log(metrics, 'train')

        return {}

    def validation_epoch_end(self, outputs):
        metrics = self.collect_metrics(outputs, 'val')
        self.__log(metrics, 'val')

        if self.eval_ema:
            metrics_ema = self.collect_metrics(outputs, 'ema')
            self.__log(metrics_ema, 'ema')
        else:
            metrics_ema = None

        if metrics.loss < self.best:
            self.best = metrics.loss

        return {
            'progress_bar': {
                'val_loss': metrics.loss,
                'best': self.best,
            },
            'val_loss': metrics.loss,
            'ema_loss': metrics_ema.loss if metrics_ema is not None else None,
        }

    @abstractmethod
    def collect_metrics(self, outputs: List[Dict], prefix: str) -> T:
        pass

    def __log(self, metrics: T, prefix: str):
        if self.global_step > 0:
            self.tb_logger.add_scalar('lr', metrics.lr, self.current_epoch)
            for k, v in dataclasses.asdict(metrics).items():
                if k == 'lr':
                    continue
                self.tb_logger.add_scalars(k, {
                    prefix: v,
                }, self.current_epoch)

    def configure_optimizers(self):
        if self.hp.optim == 'sgd':
            opt = SGD(
                self.model.parameters(),
                lr=self.hp.lr,
                momentum=0.9,
                nesterov=True,
            )
            sched = {
                'scheduler': OneCycleLR(
                    opt,
                    max_lr=self.hp.lr,
                    total_steps=self.total_steps),
                'interval': 'step',
            }
            return [opt], [sched]

        if self.hp.optim == 'ranger':
            optim = Ranger
        elif self.hp.optim == 'radam':
            optim = RAdam
        else:
            raise Exception(f'Not supported optim: {self.hp.optim}')
        opt = optim(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )
        sched = {
            'scheduler': LambdaLR(
                opt,
                lr_lambda=flat_cos(self.total_steps),
            ),
            'interval': 'step',
        }
        return [opt], [sched]

    @property
    def steps_per_epoch(self) -> int:
        return len(self.train_dataloader())

    @property
    def max_epochs(self) -> int:
        trainer: pl.Trainer = self.trainer
        return trainer.max_epochs

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.max_epochs

    @property
    def eval_ema(self) -> bool:
        if self.ema_model is None:
            return False
        f = self.hp.ema_eval_freq
        return self.current_epoch % f == f - 1

    @abstractmethod
    @cached_property
    def hp(self) -> Type[ModuleBaseParams]:
        pass

    @property
    def tb_logger(self) -> SummaryWriter:
        return self.logger.experiment
