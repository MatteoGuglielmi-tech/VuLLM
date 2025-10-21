import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


MY_PENCIL: dict[str,str] = {
    "blue": "\u001b[38;5;69m",
    "purple": "\u001b[38;5;189m",
    "yellow": "\u001b[38;5;220m",
    "red": "\u001b[38;5;160m",
    "green": "\u001b[38;5;34m",
    "bold_red": "\u001b[1m\u001b[38;5;1m",
    "reset": "\u001b[0m",
}


@dataclass
class LogFMT(logging.Formatter):
    fmt: str = '| %(levelname)-8s | %(asctime)s | %(name)s:%(lineno)d | "%(message)s"'

    # -- pre-create the formatters --
    def __post_init__(self):
        super().__init__(self.fmt) # init parent class
        self.default_formatter = logging.Formatter(self.fmt) # default formatter for fallback

        reset: str = MY_PENCIL["reset"]
        self.FORMATTERS = {
            logging.DEBUG: logging.Formatter(MY_PENCIL["blue"] + self.fmt + reset),
            logging.INFO: logging.Formatter(MY_PENCIL["purple"] + self.fmt + reset),
            logging.WARNING: logging.Formatter(MY_PENCIL["yellow"] + self.fmt + reset),
            logging.ERROR: logging.Formatter(MY_PENCIL["red"]  + self.fmt + reset),
            logging.CRITICAL: logging.Formatter(MY_PENCIL["bold_red"] + self.fmt + reset),
        }

    def format(self, record: logging.LogRecord):
        return self.FORMATTERS.get(record.levelno, self.default_formatter).format(record=record)


def setup_logger(level: int = logging.DEBUG):
    """Configures and returns a logger for the module.

    Parameters
    ----------
    level : int, optional
        The logging level threshold, by default `logging.DEBUG`.
    """

    date: str = datetime.today().strftime("%Y-%m-%d")
    time: str = datetime.now().strftime("%H-%M-%S")
    log_dir = Path(__file__).parent / f"run/{date}/{time}"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / "app.log"

    # capture warnings from `warnings` built-in Python module
    logging.captureWarnings(True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # handler for console log
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(LogFMT())

    # handler for file log
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    # File handler should use a standard, non-colored formatter
    file_formatter = logging.Formatter('| %(levelname)-8s | %(asctime)s | %(name)s:%(lineno)d | "%(message)s"')
    fh.setFormatter(file_formatter)

    # add handlers
    root_logger.addHandler(sh)
    root_logger.addHandler(fh)

    # =================================================================
    # == SILENCE NOISY LIBRARIES ==
    # =================================================================
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("bitsandbytes").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("git.util").setLevel(logging.WARNING)
    logging.getLogger("git.cmd").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("graphviz").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("browser_proc").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # =================================================================

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
