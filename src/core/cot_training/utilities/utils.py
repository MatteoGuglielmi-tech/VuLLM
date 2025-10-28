from argparse import Namespace
from typing import Any

from rich.console import Console
from rich.table import Table

from .decorators import main_process_only

_console = Console()


@main_process_only
def rich_table(
    data: dict[str, Any] | Namespace,
    title: str,
    columns: list[str],
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
    is_main_process : bool | None
        Whether to display (main process only).
        - If None (default): Auto-detects from environment
        - If True: Always display
        - If False: Never display
    """

    data = vars(data) if isinstance(data, Namespace) else data

    table = Table(show_header=True, header_style="bold blue")
    table.title = "\n" + title

    for i, c in enumerate(columns):
        table.add_column(header=c, style="italic dim" if i == 0 else None)

    for key, value in data.items():
        if isinstance(value, list):  # this assumes no nested dictoinaries
            value = [str(v) for v in value]
            table.add_row(key, *value)
        else:
            table.add_row(key, str(value))

    # display
    if pre_desc:
        _console.print(pre_desc)
    _console.print(table)
    if post_desc:
        _console.print(post_desc)


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
    print("\n")
    _console.rule(title, style=style, align="center")


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
    # Cleanup
    accelerator.wait_for_everyone()  # Synchronize all processes
    accelerator.free_memory()  # Free GPU memory

    # This should be automatic, but explicit is better
    if accelerator.state.distributed_type != "NO":
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
