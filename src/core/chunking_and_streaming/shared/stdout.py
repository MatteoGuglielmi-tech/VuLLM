import logging
from dataclasses import dataclass


MY_LOGGER_NAME = "LOGGAH"


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
    fmt:str = '| %(levelname)s | %(asctime)s | %(filename)s, lnb = %(lineno)d -> "%(message)s"'

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


def setup_logger(level: int = logging.DEBUG) -> logging.Logger:
    """Configures and returns a logger for the module.

    Parameters
    ----------
    level : int, optional
        The logging level threshold, by default `logging.DEBUG`.

    Returns
    -------
    logging.Logger
        The configured logger instance.

    Notes
    -----
    This function modifies a module-level `logger` object in place.
    Setting `propagate = False` prevents log records from being passed
    to the handlers of ancestor loggers, which is useful for avoiding
    duplicate messages in complex logging hierarchies.
    """

    # logger definition (not configured yet)
    logger = logging.getLogger(MY_LOGGER_NAME)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(LogFMT())

    logger.setLevel(level)
    logger.addHandler(sh)
    logger.propagate = False # prevents logs from going to the root logger

    return logger
