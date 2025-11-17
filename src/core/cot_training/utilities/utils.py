import os
import logging
import gc
import torch

from collections.abc import Sized as ABCSized
from typing import Any, Iterable, Literal, TypeVar, Callable
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager

from rich.pretty import Pretty
from rich.align import AlignMethod, Align
from rich.console import Console, HighlighterType, JustifyMethod, Group, OverflowMethod
from rich.style import StyleType
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TaskID,
)
from rich.text import TextType

from .decorators import main_process_only
from .detection import get_accelerator_config


T = TypeVar("T")
logger = logging.getLogger(__name__)
_console = Console()


def create_progress() -> Progress:
    return Progress(
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
    show_header: bool = True,
    expand: bool = False,
) -> Table:
    data = vars(data) if isinstance(data, Namespace) else data

    table = Table(
        show_header=show_header, header_style="bold blue", box=box, expand=expand
    )
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
    justify: JustifyMethod = "center",
    pre_desc: str | None = None,
    post_desc: str | None = None,
    allow_wrap: bool = False,
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
    _console.print(data, justify=justify, no_wrap=not allow_wrap)
    if post_desc:
        _console.print(post_desc)


def build_panel(
    tables: list[Table] | Table,
    panel_title: TextType | None = None,
    subtitle: TextType | None = None,
    border_style: StyleType = "red",
    align: AlignMethod = "center",
    panel_padding: tuple = (1, 3),
    grid_padding: tuple = (1, 5),
    panel_align: Literal["left", "center", "right"] = "center",
    layout: Literal["vertical", "horizontal"] = "horizontal",
):
    """Build table(s) in a panel.

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
    panel_align : {"left", "center", "right"}, optional
        Where to align the panel.
    layout : {"vertical", "horizontal"}
        How to arrange multiple tables.
    """
    if isinstance(tables, list):
        if layout == "vertical":
            to_render = Group(*[Align.center(table) for table in tables])
        else:
            container = Table.grid(padding=grid_padding, expand=False)
            for _ in tables:
                container.add_column()

            container.add_row(*tables)
            to_render = container
    else:
        to_render = tables

    panel = Panel.fit(
        to_render,
        title=panel_title,
        subtitle=subtitle,
        border_style=border_style,
        title_align=align,
        subtitle_align=align,
        padding=panel_padding,
    )

    if panel_align == "left":
        aligned_panel = Align.left(panel)
    elif panel_align == "right":
        aligned_panel = Align.right(panel)
    else:
        aligned_panel = Align.center(panel)
    return aligned_panel


@main_process_only
def rich_panel(
    tables: list[Table] | Table,
    panel_title: TextType | None = None,
    subtitle: TextType | None = None,
    border_style: StyleType = "red",
    align: AlignMethod = "center",
    panel_padding: tuple = (1, 3),
    grid_padding: tuple = (1, 5),
    panel_align: Literal["left", "center", "right"] = "center",
    allow_wrap: bool = False,
    layout: Literal["vertical", "horizontal"] = "horizontal",
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
    panel_align : {"left", "center", "right"}, optional
        Where to align the panel.
    allow_wrap : bool
        Whether to allow content wrapping.
    layout : {"vertical", "horizontal"}
        How to arrange multiple tables.
    """

    aligned_panel = build_panel(
        tables,
        panel_title,
        subtitle,
        border_style,
        align,
        panel_padding,
        grid_padding,
        panel_align,
        layout,
    )
    _console.print(aligned_panel, no_wrap=not allow_wrap)


@main_process_only
def rich_panels_grid(
    panels: list[Panel],
    grid_shape: tuple[int, int] | None = None,
    grid_padding: tuple = (0, 2),
    align: Literal["left", "center", "right"] = "center",
):
    """Arrange multiple panels in a grid.

    Parameters
    ----------
    panels : list[Panel]
        List of Panel objects to arrange
    grid_shape : tuple[int, int] | None
        (rows, cols). If None, auto-determines
    grid_padding : tuple
        Padding between panels
    align : {"left", "center", "right"}
        Alignment of the entire grid
    """
    if grid_shape:
        _, cols = grid_shape
    else:
        import math

        cols = math.ceil(math.sqrt(len(panels)))
        # rows = math.ceil(len(panels) / cols)

    grid = Table.grid(padding=grid_padding, expand=False)

    for _ in range(cols):
        grid.add_column()

    for i in range(0, len(panels), cols):
        row_panels = panels[i : i + cols]
        # Pad with empty strings if needed
        while len(row_panels) < cols:
            row_panels.append("") # type: ignore[reportArgumentType]
        grid.add_row(*row_panels)

    if align == "left":
        aligned = Align.left(grid)
    elif align == "right":
        aligned = Align.right(grid)
    else:
        aligned = Align.center(grid)

    _console.print(aligned)


def build_pretty_renderable(
    _object,
    highlighter: HighlighterType | None = None,
    *,
    indent_size: int = 4,
    justify: JustifyMethod | None = None,
    overflow: OverflowMethod | None = None,
    no_wrap: bool = False,
    indent_guides: bool = True,
    max_length: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    expand_all: bool = False,
    margin: int = 0,
    insert_line: bool = False,
):
    return Pretty(
        _object,
        highlighter=highlighter,
        indent_size=indent_size,
        justify=justify,
        overflow=overflow,
        no_wrap=no_wrap,
        indent_guides=indent_guides,
        max_length=max_length,
        max_string=max_string,
        max_depth=max_depth,
        expand_all=expand_all,
        margin=margin,
        insert_line=insert_line,
    )


@main_process_only
def rich_print(
    _object,
    highlighter: HighlighterType | None = None,
    *,
    console: Console | None = None,
    indent_guides: bool = True,
    max_length: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    expand_all: bool = False,
    style: str = "",
    indent_size: int = 4,
    justify: JustifyMethod | None = None,
    overflow: OverflowMethod | None = None,
    no_wrap: bool = False,
    margin: int = 0,
    insert_line: bool = False,
    is_main_process: bool | None = None,
):
    """Print with rich styling (main process only)."""

    cons = console if console is not None else _console

    if isinstance(_object, str):
        cons.print(_object, style=style)
    else:
        pretty_obj = build_pretty_renderable(
            _object, 
            highlighter=highlighter,
            indent_size=indent_size,
            justify=justify,
            overflow=overflow,
            no_wrap=no_wrap,
            indent_guides=indent_guides,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            expand_all=expand_all,
            margin=margin,
            insert_line=insert_line,
        )

        cons.print(pretty_obj)


@main_process_only
def rich_exception(is_main_process: bool | None = None, show_locals: bool=False):
    """Print exception traceback with locals (main process only).
    No-op if not in exception context.
    """

    try:
        _console.print_exception(show_locals=show_locals)
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

    _progress = create_progress()
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
    status_fn: Callable[[T], str] | None = None,
):
    """Automatic progress bar with optional dynamic status.

    Parameters
    ----------
    iterable : Iterable
        Items to iterate over
    total : int | None
        Total count. If None, tries len(iterable)
    description : str
        Description to display
    initial_status : str
        Initial status message
    status_fn : Callable[[T], str] | None
        Function that takes the current item and returns a status string
    """

    if total is None:
        if isinstance(iterable, ABCSized):
            total = len(iterable)
        else:
            raise ValueError(
                "Must provide 'total' when iterable doesn't support len(). "
                "Try: rich_progress_advanced(iterable, total=<count>)"
            )

    _progress = create_progress()
    with _progress:
        task = _progress.add_task(description=description, total=total, status=initial_status)
        for item in iterable:
            yield item
            status = status_fn(item) if status_fn else None
            _progress.update(task, advance=1, status=status)


@contextmanager
def rich_status(description: str, spinner: str = "clock"):
    """Context manager for rich status spinner."""
    with _console.status(description, spinner=spinner) as s:
        try:
            yield s
            s.update(f"{description} -> [bold green]✓ Done[/bold green]")
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


@main_process_only
def display_env_info(parser: ArgumentParser, args: Namespace):
    # env settings
    cpus_allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    accelerator = get_accelerator_config()

    data = {
        "distributed type": accelerator.state.distributed_type.name,
        "Num processes": accelerator.num_processes,
        "Mixed precision": accelerator.mixed_precision,
        "Num SLURM CPUs": cpus_allocated,
    }
    if accelerator.state.deepspeed_plugin is not None:
        data["DEEPSPEED"] = "ENABLED"
        data["ZeRO Stage"] = accelerator.state.deepspeed_plugin.zero_stage
        data["Offload optimizer"] = (
            accelerator.state.deepspeed_plugin.offload_optimizer_device
        )
        data["Offload params"] = accelerator.state.deepspeed_plugin.offload_param_device
    else:
        data["DEEPSPEED"] = "DISABLED"

    acc = build_table(
        data=data, title="Meta information", columns=["Parameter", "Value"]
    )

    # cli args
    MANDATORY_GROUPS = ["Common Arguments"]
    if args.finetune:
        MANDATORY_GROUPS.extend(
            [
                "Shared Training Arguments (Fine-tuning & HPO)",
                "Fine-tuning only arguments"
            ]
        )
        custom_args = {
            action.dest: getattr(args, action.dest, None)
            for group in parser._action_groups
            if group.title in set(MANDATORY_GROUPS)
            for action in group._group_actions
        }
    elif args.hpo:
        MANDATORY_GROUPS.extend(
            [
                "Shared Training Arguments (Fine-tuning & HPO)",
                "Hyperparameter Optimization Arguments",
            ]
        )
        custom_args = {
            action.dest: getattr(args, action.dest, None)
            for group in parser._action_groups
            if group.title in set(MANDATORY_GROUPS)
            for action in group._group_actions
        }
    else:  # merge
        MANDATORY_GROUPS.extend(["Inference Arguments"])
        custom_args = {
            action.dest: getattr(args, action.dest, None)
            for group in parser._action_groups
            if group.title in set(MANDATORY_GROUPS)
            for action in group._group_actions
        }

    cli = build_table(
        data=custom_args, title="CLI arguments", columns=["Name", "Value"]
    )
    rich_rule(style="royal_blue1")
    rich_panel(
        tables=[acc, cli],
        panel_title="Run settings",
        border_style="royal_blue1",
    )
    rich_rule()
