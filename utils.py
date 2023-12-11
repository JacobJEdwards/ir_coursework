from time import time
from functools import wraps
import logging
from typing import Callable, Sequence, ParamSpec, Any

Params = ParamSpec("Params")


def is_verbose(args: Sequence[Any]) -> bool:
    # dirty
    return any(
        hasattr(arg, "verbose")
        and isinstance(arg.verbose, bool)
        and arg.verbose is True
        for arg in args
    )


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrap(*args: Params.args, **kw: Params.kwargs) -> Any:
        if not is_verbose(args):
            return func(*args, **kw)

        ts = time()
        result = func(*args, **kw)
        te = time()
        logging.info(f"func: {func.__name__} took: {te - ts:2.4f} sec")
        return result

    return wrap
