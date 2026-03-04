import inspect
import logging

from functools import wraps
from pathlib import Path
from typing import Callable, Union, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


logger = logging.getLogger(__name__)


class InvalidFileExtensionError(ValueError):
    """Raised when a file has an invalid extension"""

    pass


def validate_extension(
    target_param_name: str,
    allowed_extensions: Union[str, list[str]],
    case_sensitive: bool = False,
):
    """
    Decorator to validate file extension of a parameter

    Args:
        param_name: Name of the parameter to validate (e.g., 'filename', 'output_fp')
        allowed_extensions: Single extension or list of allowed extensions (e.g., 'json' or ['json', 'jsonl'])
        case_sensitive: Whether to do case-sensitive comparison (default: False)

    Raises:
        InvalidFileExtensionError: If the file extension is not in allowed_extensions
        ValueError: If the parameter is not found or is not a Path-like object

    Example:
        @validate_extension('filename', ['json', 'jsonl'])
        def save_data(self, output_dir: Path, filename: Path):
            ...
    """

    if isinstance(allowed_extensions, str):
        allowed_extensions = [allowed_extensions]

    normalized_extensions: list[str] = []
    for ext in allowed_extensions:
        ext = ext if ext.startswith(".") else f".{ext}"
        normalized_extensions.append(ext if case_sensitive else ext.lower())

    def decorator(func: Callable[P, R]) -> Callable[P,R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if target_param_name not in bound_args.arguments:
                raise ValueError(
                    f"Parameter '{target_param_name}' not found in function '{func.__name__}'. "
                    f"Available parameters: {list(bound_args.arguments.keys())}"
                )

            param_value = bound_args.arguments[target_param_name]

            if isinstance(param_value, str):
                param_value = Path(param_value)
            elif not isinstance(param_value, Path):
                raise ValueError(
                    f"Parameter '{target_param_name}' must be a Path or str, "
                    f"got {type(param_value).__name__}"
                )

            file_extension = param_value.suffix
            if not case_sensitive:
                file_extension = file_extension.lower()

            if file_extension not in normalized_extensions:
                display_extensions = [ext.lstrip(".") for ext in normalized_extensions]
                raise InvalidFileExtensionError(
                    f"Invalid file extension for '{target_param_name}': got '{param_value.suffix}', "
                    f"expected one of: {display_extensions}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_json(target_param_name: str = "filename"):
    """Shorthand for JSON validation"""
    return validate_extension(target_param_name, "json")


def validate_jsonl(target_param_name: str = "filename"):
    """Shorthand for CSV validation"""
    return validate_extension(target_param_name, "jsonl")


class InvalidPathTypeError(TypeError):
    """Raised when a parameter is not a Path or str"""

    pass


class PathNotFoundError(FileNotFoundError):
    """Raised when a required path does not exist"""

    pass


def validate_path(
    target_param_name: str,
    check_existence: bool = False,
    create_if_missing: bool = False,
    must_be_dir: bool = False,
    must_be_file: bool = False,
):
    """
    Decorator to validate and optionally create Path parameters

    Parameters
    ----------
        target_param_name: Name of the parameter to validate (e.g., 'output_dir', 'input_file')
        check_existence: If True, raise error if path doesn't exist (unless create_if_missing=True)
        create_if_missing: If True, create the path if it doesn't exist (only works with directories)
        must_be_dir: If True, validate that path is a directory (when checking existence)
        must_be_file: If True, validate that path is a file (when checking existence)

    Raises
    ------
        InvalidPathTypeError: If the parameter is not a Path or str
        PathNotFoundError: If check_existence=True and path doesn't exist (and create_if_missing=False)
        ValueError: If the parameter is not found, or if must_be_dir and must_be_file are both True

    Example:
        # Validate output directory exists or create it
        @validate_path('output_dir', create_if_missing=True, must_be_dir=True)
        def save_data(self, output_dir: Path, filename: str):
            ...

        # Validate input file exists
        @validate_path('input_file', check_existence=True, must_be_file=True)
        def load_data(self, input_file: Path):
            ...
    """

    if must_be_dir and must_be_file:
        raise ValueError("Cannot specify both must_be_dir=True and must_be_file=True")

    if create_if_missing and must_be_file:
        raise ValueError(
            "Cannot use create_if_missing=True with must_be_file=True (can only create directories)"
        )

    def decorator(func: Callable[P, R]) -> Callable[P,R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if target_param_name not in bound_args.arguments:
                raise ValueError(
                    f"Parameter '{target_param_name}' not found in function '{func.__name__}'. "
                    f"Available parameters: {list(bound_args.arguments.keys())}"
                )

            param_value = bound_args.arguments[target_param_name]

            if isinstance(param_value, str):
                param_value = Path(param_value)
                bound_args.arguments[target_param_name] = param_value
            elif not isinstance(param_value, Path):
                raise InvalidPathTypeError(
                    f"Parameter '{target_param_name}' must be a Path or str, "
                    f"got {type(param_value).__name__}"
                )

            if check_existence or create_if_missing:
                path_exists = param_value.exists()

                if not path_exists:
                    if create_if_missing:
                        param_value.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {param_value}")
                    elif check_existence:
                        raise PathNotFoundError(
                            f"Path '{param_value}' for parameter '{target_param_name}' does not exist"
                        )
                else:
                    if must_be_dir and not param_value.is_dir():
                        raise PathNotFoundError(
                            f"Path '{param_value}' for parameter '{target_param_name}' "
                            f"must be a directory, but is a file"
                        )
                    if must_be_file and not param_value.is_file():
                        raise PathNotFoundError(
                            f"Path '{param_value}' for parameter '{target_param_name}' "
                            f"must be a file, but is a directory"
                        )

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


def ensure_dir(target_param_name: str = "output_dir"):
    """Shorthand: Ensure directory exists, create if missing"""
    return validate_path(target_param_name, create_if_missing=True, must_be_dir=True)


def require_file(target_param_name: str = "input_file"):
    """Shorthand: Require that file exists"""
    return validate_path(target_param_name, check_existence=True, must_be_file=True)


def require_dir(target_param_name: str = "input_dir"):
    """Shorthand: Require that directory exists"""
    return validate_path(target_param_name, check_existence=True, must_be_dir=True)
