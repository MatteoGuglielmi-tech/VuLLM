from argparse import ArgumentError
import logging
from dotenv import load_dotenv

from src.core.cot.logging_config import setup_logger
from src.core.cot.assessment.cli import get_parser, save_running_args
from src.core.cot.generation.llm_clients.gpt import AzureCoTGenerator
from src.core.cot.assessment.quality_check import QualityAssessor

setup_logger()
load_dotenv()

logger = logging.getLogger(name=__name__)

if __name__ == "__main__":
    logger.debug("🚀 Starting baseline... 🚀")

    args = get_parser().parse_args()
    save_running_args(args)

    cot_backend = None

    match args.engine_name:
        case "gpt-4.1":
            cot_backend = AzureCoTGenerator(deployment_name=args.deployment_name)
        case _:
            raise ArgumentError(argument=None, message="Invalid proprietary model specified")

    if cot_backend is None:
        raise RuntimeError("CoT assessor could not be initialized. Check model_type and engine_name arguments.")

    qa = QualityAssessor(
        deployment_name=args.deployment_name,
        client = cot_backend,
        max_completion_tokens=args.max_completion_tokens

    )
    qa.run_assessment(input_fp=args.source, output_dir=args.target, sample_size=args.sample_size)
