import sys
import logging
import time
from threading import Thread
from time import sleep

from src.core.cot.logging_config import BufferingHandler


class Loader:
    def __init__(
        self,
        desc_msg: str = "Loading...",
        end_msg: str = " ✅ Done.",
        timeout: float = 0.1,
        logger: logging.Logger|None = None,
    ) -> None:
        """
        A context manager for terminal loader animations that buffers log messages.

        Params
        ------
        desc_msg: (str, optional)
            The loader's description. Defaults to "Loading...".
        end: (str, optional)
            Final print. Defaults to " ✅ Done.".
        timeout: (float, optional)
            Sleep time between prints. Defaults to 0.1.
        logger: (logging.Logger, optional)
            The logger to use. Defaults to the root logger.
        Usage
        -----
        This class is intended to be used ONLY with a `with` statement wrapping it.
        """

        self.desc: str = desc_msg
        self.end: str = end_msg
        self.timeout: float = timeout
        self.logger: logging.Logger = logger or logging.getLogger()
        self._thread: Thread = Thread(target=self._animate, daemon=True)
        self.spinner_chars: list[str] = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done: bool = False
        self.start_time: float = 0

    def _animate(self):
        """The animation logic that runs in a separate thread."""

        i = 0
        while not self.done:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            if sys.stdout:
                sys.stdout.write(f"\r{self.desc} {char}")
                sys.stdout.flush()
            sleep(self.timeout)
            i += 1

    def __enter__(self):
        """Start the loader and swap the log handlers."""

        self.start_time = time.perf_counter()

        self.original_handlers = self.logger.handlers[:]
        self.console_handlers_to_restore = []
        self.buffer_handler = BufferingHandler()

        console_streams = [s for s in (sys.stdout, sys.stderr) if s is not None]
        if console_streams:
            for handler in self.original_handlers:
                handler_stream = getattr(handler, "stream", None)
                if handler_stream in console_streams:
                    self.console_handlers_to_restore.append(handler)
                    self.logger.removeHandler(handler)

        self.logger.addHandler(self.buffer_handler)
        if sys.stdout:
            self._thread.start()
        return self

    def __exit__(self, *args):
        """Stops animation, restores logging, and logs the final message."""

        end_time = time.perf_counter()
        duration = end_time - self.start_time

        if self._thread.is_alive():
            self.done = True
            self._thread.join()

        if sys.stdout:
            clear_line = "\r" + " " * (len(self.desc) + 2) + "\r"
            sys.stdout.write(clear_line)
            sys.stdout.flush()

        self.logger.removeHandler(self.buffer_handler)
        for handler in self.console_handlers_to_restore:
            self.logger.addHandler(handler)

        formatted_duration = time.strftime('%H:%M:%S', time.gmtime(duration))
        self.logger.info(f"{self.end.strip()} (took {formatted_duration})")

        for handler in self.console_handlers_to_restore:
            if hasattr(self.buffer_handler, "flush_to_handler"):
                self.buffer_handler.flush_to_handler(handler)

