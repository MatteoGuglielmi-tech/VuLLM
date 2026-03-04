from argparse import ArgumentError
import logging
from dotenv import load_dotenv

from src.core.cot.generation.logging_config import setup_logger
from src.core.cot.generation.cli import get_parser, save_running_args
from src.core.cot.generation.reasoner import Reasoner


setup_logger()
load_dotenv()

logger = logging.getLogger(name=__name__)

if __name__ == "__main__":
    logger.debug("🚀 Starting baseline... 🚀")

    args = get_parser().parse_args()
    save_running_args(args)

    cot_generator = None

    if args.model_type == "open_source":
        from src.core.cot.generation.llm_clients.llama import LlamaCoTGenerator
        cot_generator=LlamaCoTGenerator(
            model_name=args.model_name,
            chat_template=args.chat_template,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )

    elif args.model_type == "proprietary":
        from src.core.cot.generation.llm_clients.gpt import AzureCoTGenerator
        match args.engine_name:
            case "gpt-4.1":
                cot_generator = AzureCoTGenerator(deployment_name=args.deployment_name)
            case "gtp-5-mini":
                cot_generator = AzureCoTGenerator(deployment_name=args.deployment_name)
            case _:
                raise ArgumentError(
                    argument=None, message="Invalid proprietary model specified"
                )
    else:
        raise ValueError(f"Invalid model_type specified: {args.model_type}")

    if cot_generator is None:
        raise RuntimeError("CoT Generator could not be initialized. Check model_type and engine_name arguments.")

    cot = Reasoner(
        input_fp=args.source,
        output_fp=args.target,
        cot_generator=cot_generator,
        batch_size=args.batch_size,
        max_completion_tokens=args.max_completion_tokens
    )

    cot.run()
