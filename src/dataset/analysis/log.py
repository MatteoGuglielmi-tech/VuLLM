import logging


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    '''
    \001b[ Escape code, this is always the same
    1 = Style, 1 for normal, 0 = bold, 2 = underline, 3 = negative1, 4 = negative2.
    32 = Text colour, 32 for bright green.
    40m = Background colour, 40 is for black.
    '''

    blue = "\u001b[38;5;69m"
    purple = "\u001b[38;5;189m"
    yellow = "\u001b[38;5;220m"
    red = "\u001b[38;5;160m"
    green = "\u001b[38;5;34m"
    bold_red = "\u001b[1m\u001b[38;5;1m"
    reset = "\u001b[0m"

    # fmt = '| %(asctime)s | %(levelname)4s | %(filename)s -> %(funcName)s(): line %(lineno)s | %(message)s'
    fmt = '| %(levelname)4s | %(filename)s -> %(funcName)s(): line %(lineno)s | %(message)s'
    FORMATS = {
        logging.DEBUG: blue + fmt + reset,
        logging.INFO: purple + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(CustomFormatter())


logger.addHandler(stdout_handler)
