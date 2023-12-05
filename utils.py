from time import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrap(*args, **kw):
        if len(args) > 0 and not args[0].verbose:
            return func(*args, **kw)

        ts = time()
        result = func(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (func.__name__, te - ts))
        return result

    return wrap
