import functools
import inspect

from pathlib import Path
from typing import TypeVar, ParamSpec, Callable, Any
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


def validate_argument_value(
    arg_name: str, allowed_values: list[str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Creates a decorator to validate an argument against a list of allowed values.

    This is a decorator factory. You call it with an argument name and a list
    of allowed string values to produce a decorator. The returned decorator,
    when applied to a function, will ensure that the specified argument is
    present, is a string, and its value is one of the allowed options.

    Parameters
    ----------
    arg_name : str
        The name of the parameter in the decorated function to validate.
    allowed_values : list[str]
        A list of string values that the specified argument is allowed to have.

    Returns
    -------
    Callable[[Callable], Callable]
        A decorator that can be applied to a function to enforce the argument
        validation.

    Raises
    ------
    TypeError
        Raised by the decorator if the specified argument is not provided or
        is not a string.
    ValueError
        Raised by the decorator if the argument's value is not in the
        `allowed_values` list.
    """

    allowed_set = set(allowed_values)  # O(1) for lookups

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sig: Signature = inspect.signature(func)
            bound_args: BoundArguments = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            candidate_value: Any | None = bound_args.arguments.get(arg_name)

            if candidate_value is None:
                raise TypeError(f"'{func.__name__}' must have a '{arg_name}' argument.")
            if not isinstance(candidate_value, str):
                raise TypeError(f"The '{arg_name}' argument must be a string.")
            if candidate_value not in allowed_set:
                raise ValueError(
                    f"The '{arg_name}' argument must be one of {allowed_values}."
                )

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


def validate_filepath_extension(
    arg_name: str, allowed_suffixes: list[str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Creates a decorator to validate a filepath argument's format.

    This factory produces a decorator that validates a function argument
    (expected to be a string or Path object) by checking its file extension
    against a list of allowed suffixes.

    Parameters
    ----------
    arg_name : str
        The name of the parameter in the decorated function to validate.
    allowed_suffixes : list[str]
        A list of allowed file suffixes (e.g., ['.json', '.csv']).

    Returns
    -------
    Callable[[Callable], Callable]
        A decorator that enforces the suffix validation.
    """

    allowed_set: set[str] = set(allowed_suffixes)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sig: Signature = inspect.signature(func)
            bound_args: BoundArguments = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            arg_value: Any | None = bound_args.arguments.get(arg_name)

            if arg_value is None:
                raise TypeError(f"'{func.__name__}' must have a '{arg_name}' argument.")
            if not isinstance(arg_value, (str, Path)):
                raise TypeError(
                    f"The '{arg_name}' argument must be a string or a Path object."
                )

            # extract format
            fp_path: Path = Path(arg_value)
            if fp_path.suffix not in allowed_set:
                raise ValueError(
                    f"Format of '{arg_name}' must be one of {allowed_suffixes}."
                )

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


ensure_jsonl_extension = validate_filepath_extension(
    arg_name="filepath", allowed_suffixes=[".jsonl"]
)
