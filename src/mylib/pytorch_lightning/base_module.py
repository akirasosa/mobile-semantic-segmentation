from abc import ABC, abstractmethod
from collections import OrderedDict
from logging import Logger, getLogger
from typing import Optional, Tuple, Protocol, Sequence, Mapping, Generic, TypeVar, TypedDict, Dict, Any, \
    Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mylib.torch.ensemble.ema import update_ema


class ModuleParams(Protocol):
    ema_decay: Optional[float]
    ema_eval_freq: int
    use_ema: bool


class StepResult(TypedDict):
    loss: torch.Tensor
    n_processed: int


ValStepResult = Tuple[StepResult, Optional[StepResult]]  # val and ema
T = TypeVar('T', bound=nn.Module)


class PLBaseModule(pl.LightningModule, ABC, Generic[T]):
    model: T

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_model: Optional[T] = None
        self.n_processed: int = 0

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        result = self.step(self.model, batch)
        self.n_processed += result['n_processed']

        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.hp.ema_decay)

        return result

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        result = self.step(self.model, batch)

        if self.eval_ema:
            result_ema = self.step(self.ema_model, batch)
        else:
            result_ema = None

        return result, result_ema

    @abstractmethod
    def step(self, model: T, batch) -> StepResult:
        pass

    def training_epoch_end(self, outputs: Sequence[StepResult]):
        metrics = self.collect_metrics(outputs)
        metrics = {
            **metrics,
            'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
        }
        self.__log(metrics, prefix='train')

    def validation_epoch_end(self, outputs_list: Union[Sequence[StepResult], Sequence[Sequence[ValStepResult]]]):
        # Ensure that val loader is a list.
        if isinstance(self.val_dataloader(), DataLoader):
            outputs_list = [outputs_list]

        result: Dict[str, Any] = {}

        for idx, outputs in enumerate(outputs_list):
            val_outputs = [val for val, ema in outputs]
            metrics = self.collect_metrics(val_outputs)
            self.__log(metrics, prefix=f'val_{idx}')
            result = {
                **result,
                f'val_{idx}_loss': metrics['loss'],
            }

            ema_outputs = [ema for val, ema in outputs]
            if not any(x is None for x in ema_outputs):
                metrics_ema = self.collect_metrics(ema_outputs)
                self.__log(metrics_ema, prefix=f'ema_{idx}')
                result = {
                    **result,
                    f'ema_{idx}_loss': metrics_ema['loss'],
                }
        self.log_dict(result)

    @staticmethod
    @torch.no_grad()
    def collect_metrics(outputs: Sequence[StepResult]) -> Mapping:
        loss = 0.
        total = 0
        for x in outputs:
            total += x['n_processed']
            loss += x['loss'] * x['n_processed']
        loss /= total

        return {
            'loss': loss,
        }

    def __log(self, metrics: Mapping, prefix: str):
        if self.n_processed == 0:
            return

        for k, v in metrics.items():
            if k == 'lr':
                self.tb_logger.add_scalars('lr', {
                    'lr': metrics['lr'],
                }, self.n_processed)
            else:
                self.tb_logger.add_scalars(k, {
                    prefix: v,
                }, self.n_processed)

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

    @property
    def hp(self) -> ModuleParams:
        pass

    @property
    def tb_logger(self) -> Optional[SummaryWriter]:
        if isinstance(self.logger.experiment, SummaryWriter):
            return self.logger.experiment
        for l in self.logger:
            if isinstance(l.experiment, SummaryWriter):
                return l.experiment
        return None

    @property
    def logger_(self) -> Logger:
        return getLogger('lightning')


def load_pretrained_dict(ckpt_path: str) -> OrderedDict:
    ckpt = torch.load(ckpt_path)

    if any(k.startswith('ema_model') for k in ckpt['state_dict'].keys()):
        prefix = 'ema_model'
    else:
        prefix = 'model'

    new_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if not k.startswith(f'{prefix}.'):
            continue
        new_dict[k[len(f'{prefix}.'):]] = v

    return new_dict
