import logging

from .vulcan_pipeline import Vulcan
from ...common.logging_config import setup_logger

setup_logger()
logger = logging.getLogger(name=__name__)
 
if __name__ == "__main__":
    logger.info("Application starting...")
    pipeline = Vulcan()
    logger.debug("Initialization completed")
    pipeline.run()
