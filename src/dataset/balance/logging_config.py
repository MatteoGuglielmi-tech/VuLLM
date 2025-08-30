import logging
import sys
from dataclasses import dataclass


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

    # add handlers
    root_logger.addHandler(sh)

    # =================================================================
    # == SILENCE NOISY LIBRARIES ==
    # =================================================================
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    # =================================================================

