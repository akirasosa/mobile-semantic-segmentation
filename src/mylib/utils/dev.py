from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        t = time()
        result = f(*args, **kw)
        print(time() - t)
        return result

    return wrap
