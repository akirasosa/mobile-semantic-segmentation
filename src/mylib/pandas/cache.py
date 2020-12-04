import hashlib
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd


def _make_hash(func: Callable, args, kwargs) -> str:
    k = f'{func.__name__}_{repr(args)}_{repr(kwargs)}'
    h = hashlib.md5(k.encode('utf8')).hexdigest()[:6]
    return h


def _to_parquet(df: pd.DataFrame, path: Path):
    # Columns must be string in parquet.
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(path)


_FN_MAP = {
    '.pqt': {
        'read_fn': pd.read_parquet,
        'write_fn': _to_parquet,
    },
    '.pkl': {
        'read_fn': pd.read_pickle,
        'write_fn': lambda df, path: df.to_pickle(path)
    },
}


@dataclass
class pd_cache:
    cache_dir: Union[Path, str]
    file_name: Optional[str] = None
    ext: Optional[str] = '.pqt'
    hard_reset: bool = False

    def __post_init__(self):
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        if self.file_name is None:
            assert self.ext is not None, 'ext must be specified, when fname is None.'
        else:
            self.ext = Path(self.file_name).suffix
            assert self.ext in _FN_MAP.keys(), '.pqt or .pkl is supported as file_name.'

    def __call__(self, func):
        @wraps(func)
        def cache_function(*args, **kwargs):
            cache_path = self._make_cache_path(func, args, kwargs)

            if self.hard_reset or (not cache_path.exists()):
                val = func(*args, **kwargs)
                self._write_cache(val, cache_path)
                return val

            val = self._read_cache(cache_path)
            return val

        return cache_function

    def _make_cache_path(self, func: Callable, args, kwargs) -> Path:
        if self.file_name is None:
            h = _make_hash(func, args, kwargs)
            return self.cache_dir / f'{func.__name__}_{h}{self.ext}'
        return self.cache_dir / self.file_name

    def _write_cache(self, df: pd.DataFrame, path: Path):
        _FN_MAP[self.ext]['write_fn'](df, path)

    def _read_cache(self, path: Path):
        return _FN_MAP[self.ext]['read_fn'](path)


if __name__ == '__main__':
    @pd_cache('/tmp', ext='.pqt')
    def foo(x, y=1):
        print('run foo')
        return pd.DataFrame([
            {'a': 1},
            {'a': 2},
        ])


    print(foo(5, y=2))
