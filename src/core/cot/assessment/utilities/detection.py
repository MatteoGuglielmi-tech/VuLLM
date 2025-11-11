import os

from accelerate import Accelerator

_IS_MAIN_PROCESS: bool | None = None  # cache
_ACCELERATOR: Accelerator | None = None


def is_main_process() -> bool:
    """Get cached main process status."""
    global _IS_MAIN_PROCESS
    if _IS_MAIN_PROCESS is None:
        _IS_MAIN_PROCESS = detect_main_process()
    return _IS_MAIN_PROCESS


def detect_main_process() -> bool:
    """Auto-detect if this is the main process in distributed training.

    Tries multiple detection methods in order:
    1. Accelerate (if available)
    2. PyTorch DDP environment variables
    3. Default to True (single-GPU/CPU)

    Returns
    -------
    bool
        True if main process, False otherwise.
    """

    def check_rank(rank: str) -> bool:
        return int(rank) == 0

    try:
        from accelerate import PartialState

        return PartialState().is_main_process
    except (ImportError, ValueError):
        pass

    for rank_src in ["LOCAL_RANK", "RANK", "DEEPSPEED_RANK"]:
        rank = os.environ.get(rank_src)
        if rank is not None:
            return check_rank(rank)

    return True


def get_accelerator_config():
    global _ACCELERATOR
    if _ACCELERATOR is None:
        _ACCELERATOR = init_accelerator()
    return _ACCELERATOR


def init_accelerator():
    accelerator = Accelerator()
    return accelerator

