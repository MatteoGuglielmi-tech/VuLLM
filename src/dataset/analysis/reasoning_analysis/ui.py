import os
import time

from enum import StrEnum
from collections.abc import Sized as ABCSized
from typing import Any, Iterable, Literal, TypeVar, Callable, TypedDict, cast
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager

from rich.pretty import Pretty
from rich.align import AlignMethod, Align
from rich.console import Console, HighlighterType, JustifyMethod, Group, OverflowMethod
from rich.style import StyleType, Style
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

T = TypeVar("T")
_console = Console()


class RichColors(StrEnum):
    NONE = ""
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    WHITE = "white"
    BRIGHT_BLACK = "bright_black"
    BRIGHT_RED = "bright_red"
    BRIGHT_GREEN = "bright_green"
    BRIGHT_YELLOW = "bright_yellow"
    BRIGHT_BLUE = "bright_blue"
    BRIGHT_MAGENTA = "bright_magenta"
    BRIGHT_CYAN = "bright_cyan"
    BRIGHT_WHITE = "bright_white"
    GREY0 = "grey0"
    NAVY_BLUE = "navy_blue"
    DARK_BLUE = "dark_blue"
    BLUE3 = "blue3"
    BLUE1 = "blue1"
    DARK_GREEN = "dark_green"
    DEEP_SKY_BLUE4 = "deep_sky_blue4"
    DODGER_BLUE3 = "dodger_blue3"
    DODGER_BLUE2 = "dodger_blue2"
    GREEN4 = "green4"
    SPRING_GREEN4 = "spring_green4"
    TURQUOISE4 = "turquoise4"
    DEEP_SKY_BLUE3 = "deep_sky_blue3"
    DODGER_BLUE1 = "dodger_blue1"
    DARK_CYAN = "dark_cyan"
    LIGHT_SEA_GREEN = "light_sea_green"
    DEEP_SKY_BLUE2 = "deep_sky_blue2"
    DEEP_SKY_BLUE1 = "deep_sky_blue1"
    GREEN3 = "green3"
    SPRING_GREEN3 = "spring_green3"
    CYAN3 = "cyan3"
    DARK_TURQUOISE = "dark_turquoise"
    TURQUOISE2 = "turquoise2"
    GREEN1 = "green1"
    SPRING_GREEN2 = "spring_green2"
    SPRING_GREEN1 = "spring_green1"
    MEDIUM_SPRING_GREEN = "medium_spring_green"
    CYAN2 = "cyan2"
    CYAN1 = "cyan1"
    PURPLE4 = "purple4"
    PURPLE3 = "purple3"
    BLUE_VIOLET = "blue_violet"
    GREY37 = "grey37"
    MEDIUM_PURPLE4 = "medium_purple4"
    SLATE_BLUE3 = "slate_blue3"
    ROYAL_BLUE1 = "royal_blue1"
    CHARTREUSE4 = "chartreuse4"
    PALE_TURQUOISE4 = "pale_turquoise4"
    STEEL_BLUE = "steel_blue"
    STEEL_BLUE3 = "steel_blue3"
    CORNFLOWER_BLUE = "cornflower_blue"
    DARK_SEA_GREEN4 = "dark_sea_green4"
    CADET_BLUE = "cadet_blue"
    SKY_BLUE3 = "sky_blue3"
    CHARTREUSE3 = "chartreuse3"
    SEA_GREEN3 = "sea_green3"
    AQUAMARINE3 = "aquamarine3"
    MEDIUM_TURQUOISE = "medium_turquoise"
    STEEL_BLUE1 = "steel_blue1"
    SEA_GREEN2 = "sea_green2"
    SEA_GREEN1 = "sea_green1"
    DARK_SLATE_GRAY2 = "dark_slate_gray2"
    DARK_RED = "dark_red"
    DARK_MAGENTA = "dark_magenta"
    ORANGE4 = "orange4"
    LIGHT_PINK4 = "light_pink4"
    PLUM4 = "plum4"
    MEDIUM_PURPLE3 = "medium_purple3"
    SLATE_BLUE1 = "slate_blue1"
    WHEAT4 = "wheat4"
    GREY53 = "grey53"
    LIGHT_SLATE_GREY = "light_slate_grey"
    MEDIUM_PURPLE = "medium_purple"
    LIGHT_SLATE_BLUE = "light_slate_blue"
    YELLOW4 = "yellow4"
    DARK_SEA_GREEN = "dark_sea_green"
    LIGHT_SKY_BLUE3 = "light_sky_blue3"
    SKY_BLUE2 = "sky_blue2"
    CHARTREUSE2 = "chartreuse2"
    PALE_GREEN3 = "pale_green3"
    DARK_SLATE_GRAY3 = "dark_slate_gray3"
    SKY_BLUE1 = "sky_blue1"
    CHARTREUSE1 = "chartreuse1"
    LIGHT_GREEN = "light_green"
    AQUAMARINE1 = "aquamarine1"
    DARK_SLATE_GRAY1 = "dark_slate_gray1"
    DEEP_PINK4 = "deep_pink4"
    MEDIUM_VIOLET_RED = "medium_violet_red"
    DARK_VIOLET = "dark_violet"
    PURPLE = "purple"
    MEDIUM_ORCHID3 = "medium_orchid3"
    MEDIUM_ORCHID = "medium_orchid"
    DARK_GOLDENROD = "dark_goldenrod"
    ROSY_BROWN = "rosy_brown"
    GREY63 = "grey63"
    MEDIUM_PURPLE2 = "medium_purple2"
    MEDIUM_PURPLE1 = "medium_purple1"
    DARK_KHAKI = "dark_khaki"
    NAVAJO_WHITE3 = "navajo_white3"
    GREY69 = "grey69"
    LIGHT_STEEL_BLUE3 = "light_steel_blue3"
    LIGHT_STEEL_BLUE = "light_steel_blue"
    DARK_OLIVE_GREEN3 = "dark_olive_green3"
    DARK_SEA_GREEN3 = "dark_sea_green3"
    LIGHT_CYAN3 = "light_cyan3"
    LIGHT_SKY_BLUE1 = "light_sky_blue1"
    GREEN_YELLOW = "green_yellow"
    DARK_OLIVE_GREEN2 = "dark_olive_green2"
    PALE_GREEN1 = "pale_green1"
    DARK_SEA_GREEN2 = "dark_sea_green2"
    PALE_TURQUOISE1 = "pale_turquoise1"
    RED3 = "red3"
    DEEP_PINK3 = "deep_pink3"
    MAGENTA3 = "magenta3"
    DARK_ORANGE3 = "dark_orange3"
    INDIAN_RED = "indian_red"
    HOT_PINK3 = "hot_pink3"
    HOT_PINK2 = "hot_pink2"
    ORCHID = "orchid"
    ORANGE3 = "orange3"
    LIGHT_SALMON3 = "light_salmon3"
    LIGHT_PINK3 = "light_pink3"
    PINK3 = "pink3"
    PLUM3 = "plum3"
    VIOLET = "violet"
    GOLD3 = "gold3"
    LIGHT_GOLDENROD3 = "light_goldenrod3"
    TAN = "tan"
    MISTY_ROSE3 = "misty_rose3"
    THISTLE3 = "thistle3"
    PLUM2 = "plum2"
    YELLOW3 = "yellow3"
    KHAKI3 = "khaki3"
    LIGHT_YELLOW3 = "light_yellow3"
    GREY84 = "grey84"
    LIGHT_STEEL_BLUE1 = "light_steel_blue1"
    YELLOW2 = "yellow2"
    DARK_OLIVE_GREEN1 = "dark_olive_green1"
    DARK_SEA_GREEN1 = "dark_sea_green1"
    HONEYDEW2 = "honeydew2"
    LIGHT_CYAN1 = "light_cyan1"
    RED1 = "red1"
    DEEP_PINK2 = "deep_pink2"
    DEEP_PINK1 = "deep_pink1"
    MAGENTA2 = "magenta2"
    MAGENTA1 = "magenta1"
    ORANGE_RED1 = "orange_red1"
    INDIAN_RED1 = "indian_red1"
    HOT_PINK = "hot_pink"
    MEDIUM_ORCHID1 = "medium_orchid1"
    DARK_ORANGE = "dark_orange"
    SALMON1 = "salmon1"
    LIGHT_CORAL = "light_coral"
    PALE_VIOLET_RED1 = "pale_violet_red1"
    ORCHID2 = "orchid2"
    ORCHID1 = "orchid1"
    ORANGE1 = "orange1"
    SANDY_BROWN = "sandy_brown"
    LIGHT_SALMON1 = "light_salmon1"
    LIGHT_PINK1 = "light_pink1"
    PINK1 = "pink1"
    PLUM1 = "plum1"
    GOLD1 = "gold1"
    LIGHT_GOLDENROD2 = "light_goldenrod2"
    NAVAJO_WHITE1 = "navajo_white1"
    MISTY_ROSE1 = "misty_rose1"
    THISTLE1 = "thistle1"
    YELLOW1 = "yellow1"
    LIGHT_GOLDENROD1 = "light_goldenrod1"
    KHAKI1 = "khaki1"
    WHEAT1 = "wheat1"
    CORNSILK1 = "cornsilk1"
    GREY100 = "grey100"
    GREY3 = "grey3"
    GREY7 = "grey7"
    GREY11 = "grey11"
    GREY15 = "grey15"
    GREY19 = "grey19"
    GREY23 = "grey23"
    GREY27 = "grey27"
    GREY30 = "grey30"
    GREY35 = "grey35"
    GREY39 = "grey39"
    GREY42 = "grey42"
    GREY46 = "grey46"
    GREY50 = "grey50"
    GREY54 = "grey54"
    GREY58 = "grey58"
    GREY62 = "grey62"
    GREY66 = "grey66"
    GREY70 = "grey70"
    GREY74 = "grey74"
    GREY78 = "grey78"
    GREY82 = "grey82"
    GREY85 = "grey85"
    GREY89 = "grey89"
    GREY93 = "grey93"


class Alignment(StrEnum):
    """Alignment options for grid layout."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"

class StyleKwargs(TypedDict, total=False):
    color: RichColors | None
    bgcolor: RichColors | None

    # boolean attributes
    bold: bool | None
    dim: bool | None
    italic: bool | None
    underline: bool | None
    blink: bool | None
    blink2: bool | None
    reverse: bool | None
    conceal: bool | None
    strike: bool | None
    underline2: bool | None
    frame: bool | None
    encircle: bool | None
    overline: bool | None


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
    columns: list[str] | None = None,
    box: box.Box = box.ASCII_DOUBLE_HEAD,
    show_header: bool = True,
    expand: bool = False,
    join_list: bool = False,
) -> Table:
    """
    Build a Rich table from dictionary or Namespace.

    Parameters
    ----------
    data : dict | Namespace
        Data to display
    title : str
        Table title
    columns : list[str] | None
        Column headers. If None, uses ["Key", "Value"]
    box : box.Box
        Box style for table borders
    show_header : bool
        Whether to show column headers
    expand : bool
        Whether to expand table to full width
    join_list : bool
        If True, join list items with comma
        If False, spread lists across columns when possible

    Returns
    -------
    Table
        Configured Rich table
    """

    data = vars(data) if isinstance(data, Namespace) else data

    if columns is None:
        columns = ["Key", "Value"]

    table = Table(
        show_header=show_header,
        header_style="bold blue",
        box=box,
        expand=expand,
        title=title,
    )

    for i, col in enumerate(columns):
        table.add_column(header=col, style="italic dim" if i == 0 else None)

    num_data_columns = len(columns) - 1

    for key, value in data.items():
        if isinstance(value, list):
            list_length = len(value)

            if join_list:
                joined_value = ", ".join(str(v) for v in value)
                table.add_row(key, joined_value)

            elif list_length == num_data_columns and num_data_columns > 1:
                row_data = [key] + [str(v) for v in value]
                table.add_row(*row_data)

            else:
                formatted_list = "\n".join(str(v) for v in value)
                table.add_row(key, formatted_list)

        else:
            # Simple value - pad with empty strings for extra columns
            row_data = [key, str(value)] + [""] * (num_data_columns - 1)
            table.add_row(*row_data)

    return table

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
    panel_align: Alignment | Literal["left", "center", "right"] = Alignment.CENTER,
    layout: Literal["vertical", "horizontal"] = "horizontal",
) -> Align:
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


def rich_panels_grid(
    panels: list[Panel] | list[Align],
    grid_shape: tuple[int, int] | None = None,
    grid_padding: tuple = (0, 2),
    align: Alignment | Literal["left", "center", "right"] = Alignment.CENTER,
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
            row_panels.append("")  # type: ignore[reportArgumentType]
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


def build_rich_style_kwargs(
    *,
    color: RichColors | str | None = None,
    bgcolor: RichColors | str | None = None,

    bold: bool | None = None,
    dim: bool | None = None,
    italic: bool | None = None,
    underline: bool | None = None,
    blink: bool | None = None,
    blink2: bool | None = None,
    reverse: bool | None = None,
    conceal: bool | None = None,
    strike: bool | None = None,
    underline2: bool | None = None,
    frame: bool | None = None,
    encircle: bool | None = None,
    overline: bool | None = None,
) -> StyleKwargs:
    """Returns a StyleKwargs dictionary with autocompletion."""

    kwargs = {k: v for k, v in locals().items() if v is not None}

    return cast(StyleKwargs, kwargs)

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
    indent_size: int = 4,
    justify: JustifyMethod | None = None,
    overflow: OverflowMethod | None = None,
    no_wrap: bool = False,
    margin: int = 0,
    insert_line: bool = False,
    is_json: bool|None = None,

    color: RichColors | None = None,
    bgcolor: RichColors | None = None,

    # boolean attributes
    bold: bool | None = None,
    dim: bool | None = None,
    italic: bool | None = None,
    underline: bool | None = None,
    blink: bool | None = None,
    blink2: bool | None = None,
    reverse: bool | None = None,
    conceal: bool | None = None,
    strike: bool | None = None,
    underline2: bool | None = None,
    frame: bool | None = None,
    encircle: bool | None = None,
    overline: bool | None = None,

    is_main_process: bool | None = None,
):
    """Print with rich styling (main process only)."""

    cons = console if console is not None else _console

    if isinstance(_object, str):
        if is_json:
            cons.print_json(_object)
        else:
            style_kwargs = build_rich_style_kwargs(
                color=color,
                bgcolor=bgcolor,
                bold=bold,
                dim=dim,
                italic=italic,
                underline=underline,
                blink=blink,
                blink2=blink2,
                reverse=reverse,
                conceal=conceal,
                strike=strike,
                underline2=underline2,
                frame=frame,
                encircle=encircle,
                overline=overline
            )
            style: Style | None = Style(**style_kwargs) if style_kwargs else None
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

def rich_rule(
    title: str = "",
    style: str = "",
    is_main_process: bool | None = None,
):
    """Print a horizontal rule (main process only)."""
    print()
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


@contextmanager
def stateless_progress(description: str, spinner: str = "dots", transient: bool = True):
    """Drop-in replacement for rich_status with completion message and no progression indicator."""

    start_time = time.perf_counter()
    progress = Progress(
        SpinnerColumn(spinner),
        TextColumn("[yellow]{task.description}"),
        transient=transient,
    )

    completed: bool = False
    end_time: float = 0.

    with progress:
        task = progress.add_task(description, total=None)

        class StatusLike:
            def update(self, new_description: str):
                progress.update(task, description=f"[yellow]{new_description}")

            def stop(self):
                nonlocal completed, end_time
                completed = True
                end_time = time.perf_counter()

        status = StatusLike()
        try:
            yield status
        except Exception:
            rich_exception()
            raise

    if completed:
        duration = end_time - start_time
        formatted_duration = time.strftime("%H:%M:%S", time.gmtime(duration))
        _console.print(
            f"[bold green]✓[/bold green] {description} -> [green]Done[/green] [yellow](took {formatted_duration})[/yellow]"
        )


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


def get_env_info(parser: ArgumentParser, args: Namespace) -> list[Table]:
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

    acc: Table = build_table(
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

    cli: Table = build_table(
        data=custom_args, title="CLI arguments", columns=["Name", "Value"]
    )

    return [acc, cli]
