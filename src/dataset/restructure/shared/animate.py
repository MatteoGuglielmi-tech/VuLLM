import sys
from threading import Thread
from time import sleep


class Loader:
    def __init__( self, desc_msg: str = "Loading...", end_msg: str = "✅ Done.", timeout: float = 0.1) -> None:
        """A simple context manager to display a loader animation in the terminal.

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.


        Usage:
            with Loader(desc="Loading..."):
                time.sleep(3) # Simulate a long task
        """

        self.desc: str = desc_msg
        self.end: str = end_msg
        self.timeout: float = timeout
        self._thread: Thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        # self.steps = ['|', '/', '-', '\\']
        self.done = False

    def _animate(self):
        """The animation logic that runs in a separate thread."""

        while not self.done:
            for char in self.steps:
                if self.done:
                    break

                sys.stdout.write(f"\r{self.desc} {char}")
                sys.stdout.flush()
                # print(f"\r{self.desc} {c}", flush=True, end="")
                sleep(self.timeout)

    def __enter__(self):
        """Starts the animation thread when entering the 'with' block."""

        self._thread.start()
        return self

    def __exit__(self, *args):
        """Stops the animation and cleans up when exiting the 'with' block."""

        self.done = True
        self._thread.join()
        sys.stdout.write(f"\r{self.desc} {self.end}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    with Loader("Loading with context manager..."):
        for i in range(10):
            sleep(0.25)
