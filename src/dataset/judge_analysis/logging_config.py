import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.style import Style
from rich.theme import Theme


def setup_logger(level: int = logging.DEBUG):
    """Configures and returns a logger for the module.

    Parameters
    ----------
    level : int, optional
        The logging level threshold, by default `logging.DEBUG`.
    """

    custom_theme = Theme(
        {
            "logging.level.notset": Style(dim=True),
            "logging.level.debug": Style(color="blue"),
            "logging.level.info": Style(color="green"),
            "logging.level.warning": Style(color="yellow"),
            "logging.level.error": Style(color="red", bold=True),
            "logging.level.critical": Style(color="red", bold=True, reverse=True),
        }
    )
    console = Console(theme=custom_theme)

    logging.captureWarnings(True)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                show_level=True,
                show_path=True,
                markup=True,
            ),
        ],
        force=True,
    )
    # =================================================================
    # == SILENCE NOISY LIBRARIES ==
    # =================================================================
    for logger_name in [
        "urllib3",
        "huggingface_hub",
        "bitsandbytes",
        "transformers",
        "matplotlib",
        "fsspec",
        "filelock",
        "PIL.PngImagePlugin",
        "git.util",
        "git.cmd",
        "asyncio",
        "graphviz",
        "kaleido",
        "choreographer",
        "browser_proc",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    # =================================================================

    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("unsloth").setLevel(logging.ERROR)

    # suppress Python warnings on non-main processes
    import warnings

    warnings.filterwarnings("ignore")
