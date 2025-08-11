import functools
import inspect
from pathlib import Path
from typing import Any, Callable
from inspect import BoundArguments, Signature

from .typedef import *

import logging
from .stdout import MY_LOGGER_NAME
logger = logging.getLogger(MY_LOGGER_NAME)


def prepare_sub_chunker_args(func: Callable) -> Callable:
    """Validates parameters for a dynamic chunking function argument.

    This decorator is designed to wrap a higher-level function that accepts
    a user-provided chunking callable, `fn`. The decorator intercepts the call
    to validate that all parameters required by `fn`'s signature are available.

    Parameters
    ----------
    func : Callable
        The function to decorate. This function must accept a callable `fn`
        and its arguments `**fn_kwargs` among its parameters.

    Returns
    -------
    Callable
        The wrapped function with validation logic for the `fn` callable.

    Raises
    ------
    ValueError
        If a required parameter for the given `fn` is missing from the
        available arguments.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> ASTChunks:
        # -- bind arguments --
        sig: Signature = inspect.signature(obj=func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        chunk = bound_args.arguments['chunk']
        tokenizer = bound_args.arguments['tokenizer']
        available_tokens = bound_args.arguments['available_tokens']
        fn = bound_args.arguments['fn']
        fn_kwargs = bound_args.arguments['fn_kwargs']

        # -- prepare arguments for 'fn' --
        callable_args = {
            "chunk_metadata": {
                "context": chunk.get("context", ""),
                "text": chunk.get("text", ""),
                "context_tcount": chunk.get("context_tcount", 0),
                "text_tcount": chunk.get("text_tcount", 0),
                "nodes": chunk.get("nodes", []),
            },
            "tokenizer": tokenizer,
            "available_tokens": available_tokens,
        }
        # add any extra argument
        callable_args.update(fn_kwargs)

        # -- validate prepared arguments against function signature --
        fn_sig = inspect.signature(fn)
        fn_params: set[str] = set(fn_sig.parameters.keys())

        # Define all parameter groups in one central place for easy maintenance
        param_groups = {
            "core": {"chunk_metadata", "tokenizer", "available_tokens"},
            "ast": {"original_full_tree", "original_full_code_bytes", "parser", "c_language"},
            "context": {"partial_context_lines"},
        }

        for group_name, required_params in param_groups.items():
            if any(p in fn_params for p in required_params):
                for param in required_params:
                    if param in fn_params and param not in callable_args:
                        raise ValueError(
                            f"Chunking function '{fn.__name__}' is missing required parameter '{param}'. "
                            f"This is a '{group_name}'-specific parameter and must be provided."
                        )

        # -- call the original function with the prepared arguments --
        bound_args.arguments.update(callable_args)

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def validate_argument_value(arg_name: str, allowed_values: list[str]) -> Callable[[Callable], Callable]:
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

    allowed_set = set(allowed_values) # O(1) for lookups
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig:Signature = inspect.signature(func)
            bound_args:BoundArguments = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            candidate_value:Any|None = bound_args.arguments.get(arg_name)

            if candidate_value is None:
                raise TypeError(f"'{func.__name__}' must have a '{arg_name}' argument.")
            if not isinstance(candidate_value, str):
                raise TypeError(f"The '{arg_name}' argument must be a string.")
            if candidate_value not in allowed_set:
                raise ValueError(f"The '{arg_name}' argument must be one of {allowed_values}.")

            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def validate_filepath_extension(arg_name: str, allowed_suffixes: list[str]) -> Callable[[Callable], Callable]:
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
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig:Signature = inspect.signature(func)
            bound_args:BoundArguments = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            arg_value:Any|None = bound_args.arguments.get(arg_name)

            if arg_value is None: raise TypeError(f"'{func.__name__}' must have a '{arg_name}' argument.")
            if not isinstance(arg_value, (str, Path)): raise TypeError(f"The '{arg_name}' argument must be a string or a Path object.")

            # extract format
            fp_path: Path = Path(arg_value)
            if fp_path.suffix not in allowed_set:
                raise ValueError(f"Format of '{arg_name}' must be one of {allowed_suffixes}.")

            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


ensure_jsonl_extension = validate_filepath_extension(arg_name="filepath", allowed_suffixes=[".jsonl"])
