import logging


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    """
    \001b[ Escape code, this is always the same
    1 = Style, 1 for normal, 0 = bold, 2 = underline, 3 = negative1, 4 = negative2.
    32 = Text colour, 32 for bright green.
    40m = Background colour, 40 is for black.
    """

    blue = "\u001b[38;5;69m"
    purple = "\u001b[38;5;189m"
    yellow = "\u001b[38;5;220m"
    red = "\u001b[38;5;160m"
    green = "\u001b[38;5;34m"
    bold_red = "\u001b[1m\u001b[38;5;1m"
    reset = "\u001b[0m"

    fmt = '| %(levelname)s | %(asctime)s | %(filename)s, lnb = %(lineno)d -> "%(message)s"'

    FORMATS = {
        logging.DEBUG: blue + fmt + reset,
        logging.INFO: purple + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# ==== EXPORTER GLOBAL VARIABLES ====
# File handler config
fh = logging.FileHandler(filename="app.log", mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(
    fmt=logging.Formatter(
        fmt='| %(levelname)s | %(asctime)s | %(filename)s, lnb = %(lineno)d -> "%(message)s"'
    )
)
# --------------------------------------

# Stream handler config
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(CustomFormatter())
# --------------------------------------

# Logger config
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)
logger.addHandler(fh)
