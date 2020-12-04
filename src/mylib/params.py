from argparse import ArgumentParser
from pathlib import Path
from typing import TypeVar, Type, Optional, Union, IO, Dict

import yaml
from dacite import from_dict
from omegaconf import OmegaConf, DictConfig

T = TypeVar('T')


class ParamsMixIn:
    @classmethod
    def load(cls: Type[T], file: Optional[str] = None) -> T:
        if file is None:
            parser = ArgumentParser()
            parser.add_argument('file', type=str)
            file = parser.parse_args().file
        data = OmegaConf.to_container(OmegaConf.load(file))

        return from_dict(data_class=cls, data=data)

    @classmethod
    def from_dict(cls: Type[T], d: Dict) -> T:
        d = OmegaConf.to_container(OmegaConf.create(d))
        return from_dict(data_class=cls, data=d)

    def pretty(self) -> str:
        return OmegaConf.to_yaml(self.dict_config())

    def dict_config(self) -> DictConfig:
        return OmegaConf.structured(self)

    def to_dict(self) -> Dict:
        return yaml.safe_load(self.pretty())

    def save(self, f: Union[str, Path, IO[str]]):
        OmegaConf.save(self.dict_config(), f)
