from itertools import cycle
from threading import Thread
from time import sleep


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """A loader-like context manager.

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)

        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    # for calling loader in with statement
    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.stop()

    def stop(self):
        self.done = True
        print(f"\n{self.end}")


if __name__ == "__main__":
    with Loader("Loading with context manager..."):
        for i in range(10):
            sleep(0.25)

    loader = Loader("Loading with object...", "That was fast!", 0.05).start()
    for i in range(10):
        sleep(0.25)
    loader.stop()
