import functools
import inspect 

from typing import Callable
from inspect import Signature, BoundArguments

from .detection import is_main_process


def main_process_only(func: Callable) -> Callable:
    """Decorator to only execute function on main process.

    Automatically detects main process, but allows override via
    `is_main_process` parameter.

    Parameters
    ----------
    func : Callable
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function that only executes on main process.

    Examples
    --------
    >>> @main_process_only
    ... def log_message(msg: str):
    ...     print(msg)

    >>> # Auto-detects main process
    >>> log_message("Hello")  # Only prints on GPU 0

    >>> # Override detection
    >>> log_message("Debug", is_main_process=True)  # Always prints
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        sig: Signature = inspect.signature(func)
        bound_args: BoundArguments = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # check if explicit
        is_main: bool | None = bound_args.arguments.get("is_main_process")
        if is_main is None: # auto-detect and cache
            is_main = is_main_process()

        if not is_main:
            return None

        if "is_main_process" in bound_args.arguments:
            del bound_args.arguments["is_main_process"]

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper
