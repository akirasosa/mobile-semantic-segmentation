from pathlib import Path
from typing import Union, Callable, Any

import pandas as pd


class PandasCache:
    def __init__(self, cache_path: Union[Path, str]):
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, fn: Callable[[Any], pd.DataFrame]):
        def inner(*args, **kwargs):
            cache_key = f'{fn.__name__}|{args}|{kwargs}'.replace('/', "\\")
            cache = self.cache_path / cache_key
            if cache.exists():
                return pd.read_parquet(cache)

            cache_val = fn(*args, **kwargs)
            cache_val.to_parquet(self.cache_path / cache_key)

            return cache_val

        return inner

    def clear(self, name: str = ''):
        files = self.cache_path.glob(f'{name}*')
        for f in files:
            f.unlink()


if __name__ == '__main__':
    c = PandasCache('/tmp/cache')


    @c
    def foo(x, y=1):
        print('run foo')
        return pd.DataFrame([
            {'a': 1},
            {'a': 2},
        ])


    foo(5, y=2)
