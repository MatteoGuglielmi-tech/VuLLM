import functools
import inspect 
import time

from typing import Callable, ParamSpec, TypeVar
from inspect import Signature, BoundArguments

from .detection import is_main_process


P = ParamSpec("P")
R = TypeVar("R")


def main_process_only(func: Callable[P, R]) -> Callable[P, R | None]:
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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        sig: Signature = inspect.signature(func)
        bound_args: BoundArguments = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # check if explicit
        is_main: bool | None = bound_args.arguments.get("is_main_process")
        if is_main is None:  # auto-detect and cache
            is_main = is_main_process()

        if not is_main:
            return None

        if "is_main_process" in bound_args.arguments:
            del bound_args.arguments["is_main_process"]

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def with_retries(max_attempts: int = 3, delay: float = 5.0):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))

            raise last_error # type: ignore[reportGeneralTypeIssues]

        return wrapper

    return decorator
