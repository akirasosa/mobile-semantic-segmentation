from typing import List

from lightgbm import Booster


class ModelExtractionCallback(object):
    def __init__(self):
        self._model = None

    def __call__(self, env):
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        return self._model

    @property
    def raw_boosters(self) -> List[Booster]:
        self._assert_called_cb()
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        return self._model.best_iteration
