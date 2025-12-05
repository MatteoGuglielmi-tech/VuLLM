import logging
import gc
import torch

from .ui import rich_status


logger = logging.getLogger(__name__)


def cleanup_resources(accelerator):
    """
    Cleanup with Accelerate support.
    """
    logger.info("Cleaning up resources...")

    try:
        # Synchronize all processes
        accelerator.wait_for_everyone()
        logger.info("✓ Processes synchronized")
    except Exception:
        logger.exception(f"During waiting for everyone to align, an error has occured")

    try:
        # Free GPU memory
        accelerator.free_memory()
        logger.info("✓ GPU memory freed")
    except Exception:
        logger.exception(f"While freeing memory, an error has occured")

    try:
        # Destroy distributed process group
        if accelerator.state.distributed_type != "NO":
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
                logger.info("✓ Distributed process group destroyed")
    except Exception:
        logger.exception(f"During distribution cleanup, an error has occured")

    try:
        # Additional CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("✓ CUDA cache cleared")
    except Exception:
        logger.exception(f"An error has occured during CUDA cleanup")

    gc.collect()
    logger.info("✓ Cleanup complete")


def cleanup_single_gpu():
    """
    Cleanup for single-GPU jobs (without Accelerate).
    """

    with rich_status("Cleaning up resources...", spinner="arc"):
        try:
            # Check if distributed is initialized
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                logger.info("Destroying distributed process group...")
                torch.distributed.destroy_process_group()
                logger.info("✓ Distributed cleaned up")
        except Exception as e:
            logger.debug(f"Distributed cleanup: {e}")

        try:
            # CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("✓ CUDA cleared")
        except Exception as e:
            logger.debug(f"CUDA cleanup: {e}")

        gc.collect()


