import json
import logging
import gc
import torch

from collections.abc import Generator
from collections.abc import Sized as ABCSized
from typing import Any, Iterable, Literal, TypeVar
from argparse import ArgumentParser, Namespace
from pathlib import Path
from contextlib import contextmanager

from rich.align import AlignMethod, Align
from rich.console import Console, JustifyMethod, Group
from rich.style import StyleType
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TaskID
)
from rich.text import TextType

from .decorators import main_process_only
from ..datatypes import ReasoningSample


T = TypeVar('T')
logger = logging.getLogger(__name__)
_console = Console()
_progress = Progress(
    SpinnerColumn(spinner_name="arc"),
    TextColumn("{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("•"),
    TextColumn("{task.completed}/{task.total}"),
    TextColumn("•"),
    TimeRemainingColumn(elapsed_when_finished=True),
    TextColumn("•"),
    TextColumn("[cyan]{task.fields[status]}"),  # Custom field
) 


def build_table(
    data: dict[str, Any] | Namespace,
    title: str = "",
    columns: list[str] = [],
    box=box.ASCII_DOUBLE_HEAD,
    show_header: bool=True,
    expand: bool = False,
) -> Table:
    data = vars(data) if isinstance(data, Namespace) else data

    table = Table(show_header=show_header, header_style="bold blue", box=box, expand=expand)
    table.title = title

    for i, c in enumerate(columns):
        table.add_column(header=c, style="italic dim" if i == 0 else None)

    for key, value in data.items():
        if isinstance(value, list):  # this assumes no nested dict
            value = [str(v) for v in value]
            table.add_row(key, *value)
        else:
            table.add_row(key, str(value))

    return table


@main_process_only
def rich_table(
    data: dict[str, Any] | Namespace | Table,
    title: str = "",
    columns: list[str] = [],
    pre_desc: str | None = None,
    post_desc: str | None = None,
    is_main_process: bool | None = None,
):
    """Display a rich table with data.

    Parameters
    ----------
    data : dict[str, Any] | Namespace
        Data to display in the table.
    title : str
        Table title.
    columns : list[str]
        Column headers.
    pre_desc : str | None
        Text to display before the table.
    post_desc : str | None
        Text to display after the table.
    release_data : bool, optional
        Whether to delete data from memory when table is plotted (only for print purposes)
    is_main_process : bool | None
        Whether to display (main process only).
        - If None (default): Auto-detects from environment
        - If True: Always display
        - If False: Never display
    """

    if not isinstance(data, Table):
        data = build_table(data, title, columns)

    if pre_desc:
        _console.print(pre_desc)
    _console.print(data, justify="center", no_wrap=True)
    if post_desc:
        _console.print(post_desc)


@main_process_only
def rich_panel(
    tables: list[Table] | Table,
    panel_title: TextType | None = None,
    subtitle: TextType | None = None,
    border_style: StyleType = "red",
    align: AlignMethod = "center",
    padding: tuple = (0, 1), 
    justify: JustifyMethod | None = "center",
    allow_wrap: bool = False,
    layout: Literal["vertical", "horizontal"] = "horizontal"
):
    """Display table(s) in a centered panel.
    
    Parameters
    ----------
    tables : list[Table] | Table
        Single table or list of tables to display.
    panel_title : str, optional
        Title for the panel.
    subtitle : str, optional
        Subtitle for the panel.
    border_style : str
        Border color/style.
    align : {"left", "center", "right"}
        Alignment for panel title/subtitle.
    padding : tuple
        Padding inside the panel (vertical, horizontal).
    justify : {"default", "left", "center", "right", "full"}, optional
        Justification for the panel itself.
    allow_wrap : bool
        Whether to allow content wrapping.
    layout : {"vertical", "horizontal"}
        How to arrange multiple tables.
    """
    if isinstance(tables, list):
        if layout == "vertical":
            centered_tables = [Align.center(table) for table in tables]
            to_render = Group(*centered_tables)
        else:
            to_render = Columns(tables, align="center", expand=True)
    else:
        to_render = tables

    panel = Panel.fit(
        to_render,
        title=panel_title,
        subtitle=subtitle,
        border_style=border_style,
        title_align=align,
        subtitle_align=align,
        padding=padding,
    )
    _console.print(panel, justify=justify, no_wrap=not allow_wrap)

@main_process_only
def rich_print(
    message: str,
    style: str = "",
    is_main_process: bool | None = None,
):
    """Print with rich styling (main process only)."""
    _console.print(message, style=style)


@main_process_only
def rich_exception(is_main_process: bool | None = None):
    """Print exception traceback with locals (main process only).
    No-op if not in exception context.
    """

    try:
        _console.print_exception(show_locals=True)
    except ValueError:
        # Not in an exception context
        _console.print(
            "[yellow]Warning: rich_exception called outside except block[/yellow]"
        )


@main_process_only
def rich_rule(
    title: str = "",
    style: str = "",
    is_main_process: bool | None = None,
):
    """Print a horizontal rule (main process only)."""
    # print("\n")
    _console.rule(title, style=style, align="center")


@main_process_only 
@contextmanager
def progress_bar(iterable: Iterable, description="Working ..."):
    with _progress:
        yield _progress.track(iterable, description=description)


@contextmanager
def rich_progress_manual(
    total: int,
    description: str = "Processing",
    initial_status: str = "Starting...",
):
    """Manual progress bar (like tqdm context manager).

    Parameters
    ----------
    total : int
        Total number of items (required)
    description : str
        Description to display
    initial_status : str
        Initial status message
    """

    class ProgressController:
        def __init__(self, progress: Progress, task: TaskID):
            self.progress = progress
            self.task = task

        def update(self, advance: int = 1):
            self.progress.update(self.task, advance=advance)

        def set_postfix(self, postfix: dict[str, Any]):
            status = ", ".join(f"{k}={v}" for k, v in postfix.items())
            self.progress.update(self.task, status=status)

        def set_description(self, description: str):
            self.progress.update(self.task, description=description)

    with _progress as progress:
        task = progress.add_task(
            description=description, total=total, status=initial_status
        )
        controller = ProgressController(progress, task)
        yield controller


def rich_progress(
    iterable: Iterable[T],
    total: int | None = None,
    description: str = "Processing",
    initial_status: str = "Starting...",
    running_status: str = "Running...",
):
    """Automatic progress bar (iterator wrapper).

    Parameters
    ----------
    iterable : Iterable
        Items to iterate over
    total : int | None
        Total count. If None, tries len(iterable)
    description : str
        Description to display
    """

    if total is None:
        # Check if iterable is Sized (has __len__)
        if isinstance(iterable, ABCSized):
            total = len(iterable)
        else:
            raise ValueError(
                "Must provide 'total' when iterable doesn't support len(). "
                "Try: rich_progress_advanced(iterable, total=<count>)"
            )

    with _progress as progress:
        task = progress.add_task(
            description=description, total=total, status=initial_status
        )
        for item in iterable:
            yield item
            progress.update(task, advance=1, status=running_status)


@contextmanager
def rich_status(description: str, spinner: str = "clock"):
    """Context manager for rich status spinner."""
    with _console.status(description, spinner=spinner) as s:
        try:
            yield s
            s.update(f"{description} [bold green]✓ Done[/bold green]")
        except Exception:
            s.stop()
            rich_exception()
            raise


def get_instruction_response_parts(tokenizer) -> tuple[str, str]:
    """Automatically extract instruction and response parts from chat template.

    Returns
    -------
    tuple[str, str]
        (instruction_part, response_part)
    """

    chat_template = tokenizer.chat_template

    if not chat_template:
        raise ValueError("Tokenizer has no chat template!")

    patterns = {
        # Llama 3+
        "llama": (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        # Qwen 2.5
        "qwen": (
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
        ),
        # ChatML (generic)
        "chatml": (
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
        ),
        # Mistral
        "mistral": (
            "[INST]",
            "[/INST]",
        ),
        # Zephyr
        "zephyr": (
            "<|user|>\n",
            "<|assistant|>\n",
        ),
    }

    # Detect based on template content
    template_lower = chat_template.lower()

    if "start_header_id" in template_lower:
        return patterns["llama"]
    elif "im_start" in template_lower or "qwen" in template_lower:
        return patterns["qwen"]
    elif "[inst]" in template_lower:
        return patterns["mistral"]
    elif "<|user|>" in template_lower:
        return patterns["zephyr"]
    else:
        raise ValueError(
            f"Could not detect chat template format. "
            f"Please specify instruction_part and response_part manually.\n"
            f"Template: {chat_template[:200]}"
        )


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
    logger.info("Cleaning up resources...")

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
    logger.info("✓ Cleanup complete")


def iter_jsonl_samples(jsonl_path: Path) -> Generator[ReasoningSample, None, None]:
    """Iterate over JSONL file, yielding ReasoningSample objects."""

    with open(file=jsonl_path, mode="r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())

                yield ReasoningSample(
                    project=data["project"],
                    cwe=data["cwe"],
                    target=data["target"],
                    func=data["func"],
                    cwe_desc=data["cwe_desc"],
                    reasoning=data["reasoning"],
                    sample_id=data.get("sample_id", f"sample_{idx}"),
                )

            except (json.JSONDecodeError, KeyError):
                logger.exception(f"Error parsing line {idx}", stack_info=True)
                continue


def setup_paths(parser: ArgumentParser) -> dict[str, Path] | None:
    """Display custom CLI arguments in a formatted table."""
    args = parser.parse_args()

    BUILTIN_GROUPS = {"positional arguments", "options"}

    custom_args = {
        action.dest: getattr(args, action.dest, None)
        for group in parser._action_groups
        if group.title not in BUILTIN_GROUPS
        for action in group._group_actions
    }

    tab = build_table(data=custom_args, columns=["Name", "Value"])
    rich_panel(tab, panel_title= "CLI Arguments", border_style="royal_blue1")
    rich_rule()
 
    if args.ensemble or args.merge:
        output_folder: Path = custom_args["output"] # type: ignore
        output_folder.mkdir(parents=True, exist_ok=True)

        return {
            "filtered": output_folder / "best_reasonings.jsonl",
            "rejected": output_folder / "rejected_reasonings.jsonl",
            "metadata": output_folder / "judge_config.json",
            "filtering_stats": output_folder / "filtering_stats.json",
        }
    elif args.sequential:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)

