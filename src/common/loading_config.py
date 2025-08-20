import sys
import logging
from threading import Thread
from time import sleep

from .logging_config import BufferingHandler


class Loader:
    def __init__( self, desc_msg: str = "Loading...", end_msg: str = " ✅ Done.", timeout: float = 0.1) -> None:
        """A simple context manager to display a loader animation in the terminal.

        Params
        ------
        desc: (str, optional)
            The loader's description. Defaults to "Loading...".
        end: (str, optional)
            Final print. Defaults to " ✅ Done.".
        timeout: (float, optional)
            Sleep time between prints. Defaults to 0.1.

        Usage
        -----
        This class is intended to be used ONLY with a `with` statement wrapping it.
        """

        self.desc: str = desc_msg
        self.end: str = end_msg
        self.timeout: float = timeout
        self._thread: Thread = Thread(target=self._animate, daemon=True)
        self.spinner_chars = [ "⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿" ]
        self.done = False

    def _animate(self):
        """The animation logic that runs in a separate thread."""

        i = 0
        num_spinners = len(self.spinner_chars)
        while not self.done:
            char = self.spinner_chars[i % num_spinners]
            if sys.stdout:
                sys.stdout.write(f"\r{self.desc} {char}")
                sys.stdout.flush()
            sleep(self.timeout)
            i += 1

    def __enter__(self):
        """Start the loader and swap the log handlers."""
        self.root_logger = logging.getLogger()
        self.original_handlers = self.root_logger.handlers[:]
        self.console_handlers_to_restore = []
        self.buffer_handler = BufferingHandler()

        # Safely build a tuple of available console stream types
        console_streams = [s for s in (sys.stdout, sys.stderr) if s is not None]

        if console_streams:
            for handler in self.original_handlers:
                handler_stream = getattr(handler, 'stream', None)
                if handler_stream in console_streams:
                    self.console_handlers_to_restore.append(handler)
                    self.root_logger.removeHandler(handler)

        self.root_logger.addHandler(self.buffer_handler)
        if sys.stdout: self._thread.start()

        return self

    def __exit__(self, *args):
        """Stops the animation and cleans up when exiting the 'with' block."""

        if self._thread.is_alive():
            self.done = True
            self._thread.join()
        if sys.stdout:
            sys.stdout.write(f"\r{self.desc} -> {self.end}\n")
            sys.stdout.flush()

        self.root_logger.removeHandler(self.buffer_handler)
        for handler in self.console_handlers_to_restore:
            self.root_logger.addHandler(handler)
            self.buffer_handler.flush_to_handler(handler)

