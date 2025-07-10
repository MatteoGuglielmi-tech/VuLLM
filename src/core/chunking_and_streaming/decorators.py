import functools
import inspect
import os
from inspect import BoundArguments, Signature
from typing import Any, Callable, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from tree_sitter_language_pack import SupportedLanguage

from .stdout import logger
from .typedef import *


# <---- Decorator for validating filepath extension when calling save_outputs ---->
def validate_filepath_extension(
    func: Callable[..., None],
) -> Callable[[Windows, str], None]:
    """A decorator that validates the 'filepath' argument of the decorated
    function.

    It checks if the file extension for 'filepath' is either '.jsonl' or
    '.csv'. If the extension doesn't match these, it falls back to
    '.jsonl' and issues a warning.

    Note: this wrapper will raise a `Expected 2 more positional arguments Pyright` linting error.
        This is fake, eveything works fine.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        sig: Signature = inspect.signature(obj=func)
        bound_args: BoundArguments = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        original_filepath: Any | None = bound_args.arguments.get("filepath")

        if original_filepath is None:
            raise TypeError(
                f"The decorated function '{func.__name__}' must have a "
                "'filepath' argument."
            )
        if not isinstance(original_filepath, str):
            raise TypeError(
                f"The 'filepath' argument for '{func.__name__}' must be a string."
            )

        base, ext = os.path.splitext(original_filepath)
        new_filepath: str = original_filepath

        if ext.lower() not in [".jsonl", ".csv"]:
            # fallback to .jsonl
            new_filepath = base + ".jsonl"
            logger.warning(
                msg=f"Warning: Invalid file extension '{ext}' for filepath "
                f"'{original_filepath}'. Falling back to '.jsonl'. "
                f"New filepath: {new_filepath}"
            )

        # update 'filepath' argument in the bound arguments
        bound_args.arguments["filepath"] = new_filepath

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


# <---- Decorator for validating chunking function parameters ---->
def validate_chunking_fn_params(
    func: Callable[..., ASTChunks],
) -> Callable[..., ASTChunks]:
    """A decorator to validate the parameters passed to the dynamic chunking
    function (fn).

    It checks if the 'fn' callable has all required parameters based on
    its signature and the provided fn_kwargs.
    """

    @functools.wraps(func)
    def wrapper(
        chunk: SubChunk,
        tokenizer: PreTrainedTokenizer,
        available_tokens: int,
        function_signature: str,
        label: str,
        lang: SupportedLanguage,
        fn: Callable[..., list[Union[LBChunk, ASTChunk]]],
        **fn_kwargs: Any,
    ) -> ASTChunks:

        # Prepare arguments to pass to the dynamic chunking function 'fn'
        callable_args = {
            "chunk_metadata": {
                "context": chunk.get("context", ""),
                "text": chunk.get("text", ""),
                "context_tcount": len(
                    tokenizer.encode(
                        text=chunk.get("context", ""), add_special_tokens=False
                    )
                ),
                "text_tcount": len(
                    tokenizer.encode(
                        text=chunk.get("text", ""), add_special_tokens=False
                    )
                ),
                "nodes": chunk.get("nodes", []),
            },
            "tokenizer": tokenizer,
            "available_tokens": available_tokens,
        }

        callable_args.update(fn_kwargs)

        sig: Signature = inspect.signature(obj=fn)
        fn_params: set[str] = set(sig.parameters.keys())

        # check for core parameters
        required_params: set[str] = {
            "chunk_metadata",
            "tokenizer",
            "available_tokens",
        }

        for param in required_params:
            if param in fn_params and param not in callable_args:
                raise ValueError(
                    f"Missing required parameter '{param}' for chunking function '{fn.__name__}'."
                    f" Ensure it's correctly passed by semantic_sliding_window or via fn_kwargs."
                )

        # check for parameters specific to AST-based chunking
        ast_specific_params: set[str] = {
            "original_full_tree",
            "original_full_code_bytes",
            "parser",
            "c_language",
        }

        if any([p in fn_params for p in ast_specific_params]):
            for param in ast_specific_params:
                if param in fn_params and param not in callable_args:
                    raise ValueError(
                        f"Chunking function '{fn.__name__}' requires AST-related parameter '{param}',"
                        f" but it was not provided in fn_kwargs. This is likely an AST-based function."
                    )

        # check for 'partial_context_lines' which is often used by both types but might be in kwargs
        if (
            "partial_context_lines" in fn_params
            and "partial_context_lines" not in callable_args
        ):
            raise ValueError(
                f"Chunking function '{fn.__name__}' requires 'partial_context_lines',"
                f" but it was not provided in fn_kwargs. Please specify it."
            )

        # call the wrapped function (semantic_sliding_window) with its arguments
        # and pass the validated callable_args to the 'fn' inside it.
        return func(
            chunk=chunk,
            tokenizer=tokenizer,
            available_tokens=available_tokens,
            function_signature=function_signature,
            label=label,
            lang=lang,
            fn=fn,
            _callable_args=callable_args,
            **fn_kwargs,
        )

    return wrapper


def safeguard_label_values(func: Callable[..., Windows]) -> Callable[..., Windows]:
    """A decorator that validates the value for 'label' argument of the
    decorated function.

    It checks if the label argument of the wrapped function is either
    "0" or "1"
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Windows:
        sig: Signature = inspect.signature(obj=func)
        bound_args: BoundArguments = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        candidate_label: Any | None = bound_args.arguments.get("label", None)

        if not candidate_label:
            raise TypeError(f"'{func.__name__}' must have a 'label' argument.")
        if not isinstance(candidate_label, str):
            raise TypeError(
                f"The 'label' argument for '{func.__name__}' must be a string."
            )
        if not candidate_label in ["0", "1"]:
            raise ValueError(
                f"The 'label' argument for '{func.__name__}' must be either `0` or `1`."
            )

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def safeguard_trimming_type(func: Callable[..., Windows]) -> Callable[..., Windows]:
    """A decorator that validates the value for 'trimming_technique' argument
    of the decorated function.

    It checks if the  argument of the wrapped function is either "ast"
    or "line"
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Windows:
        sig: Signature = inspect.signature(obj=func)
        bound_args: BoundArguments = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        candidate_type: Any | None = bound_args.arguments.get(
            "trimming_technique", None
        )

        if not candidate_type:
            raise TypeError(
                f"'{func.__name__}' must have a 'trimming_technique' argument."
            )
        if not isinstance(candidate_type, str):
            raise TypeError(
                f"The 'trimming_technique' argument for '{func.__name__}' must be a string."
            )
        if not candidate_type in ["ast", "line"]:
            raise ValueError(
                f"The 'trimming_technique' argument for '{func.__name__}' must be either `ast` or `line`."
            )

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper
