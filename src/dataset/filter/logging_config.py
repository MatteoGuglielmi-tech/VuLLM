import sys
import logging

from pathlib import Path
from datetime import datetime
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

    date: str = datetime.today().strftime("%Y-%m-%d")
    time: str = datetime.now().strftime("%H-%M-%S")
    log_file = Path(__file__).parent / f"run/{date}/{time}/app.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)

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

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('| %(levelname)-8s | %(asctime)s | %(name)s:%(lineno)d | "%(message)s"')
    fh.setFormatter(file_formatter)

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
            fh
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

    logging.info(f"Logging configured. Outputting to console and '{log_file}'.")


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
