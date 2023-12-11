from time import time
from functools import wraps
import logging
from typing import Callable, Sequence, ParamSpec, Any

Params = ParamSpec("Params")


def is_verbose(args: Sequence[Any]) -> bool:
    """
    Checks if any of the arguments have a 'verbose' attribute set to True.

    Args:
        args (Sequence[Any]): The arguments passed to the function.

    Returns:
        bool: True if any argument has 'verbose' attribute set to True, False otherwise.
    """
    return any(
        hasattr(arg, "verbose")
        and isinstance(arg.verbose, bool)
        and arg.verbose is True
        for arg in args
    )


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to measure the execution time of a function.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The decorated function.
    """

    @wraps(func)
    def wrap(*args: Params.args, **kw: Params.kwargs) -> Any:
        """
        The wrapped function that measures execution time if 'verbose' is True in arguments.

        Args:
            *args: Positional arguments passed to the function.
            **kw: Keyword arguments passed to the function.

        Returns:
            Any: The result of the function call.
        """
        if not is_verbose(args):
            return func(*args, **kw)

        ts = time()
        result = func(*args, **kw)
        te = time()
        logging.info(f"func: {func.__name__} took: {te - ts:2.4f} sec")
        return result

    return wrap
