import sys
import logging

from rich.logging import RichHandler
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

from .utilities import is_main_process


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

    main_process = is_main_process()
    if main_process:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            # datefmt="[%X]",
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

    if not main_process:
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.getLogger("unsloth").setLevel(logging.ERROR)

        # suppress Python warnings on non-main processes
        import warnings
        warnings.filterwarnings("ignore")


class BufferingHandler(logging.Handler):
    """A logging handler that stores messages in a buffer and flushes them later."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []

    def emit(self, record):
        """Append the record to the buffer."""

        self.buffer.append(record)

    def flush_to_handler(self, handler):
        """Format and emit all buffered records to a target handler."""

        if not self.buffer: return

        console_streams = [s for s in (sys.stdout, sys.stderr) if s is not None]
        handler_stream = getattr(handler, 'stream', None)
        if handler_stream in console_streams: handler.stream.write("\n")
        for record in self.buffer: handler.emit(record)
        self.buffer.clear()
