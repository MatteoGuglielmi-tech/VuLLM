from typing import Literal, overload
from dataclasses import dataclass


@dataclass(frozen=True)
class PlotlyColorPalette:
    """Color palette for Plotly charts with fill and line colors."""

    fill: list[str]
    line: list[str]

    def __len__(self) -> int:
        return len(self.fill)

    def __getitem__(self, index: int) -> tuple[str, str]:
        """Get (fill_color, line_color) tuple at index."""
        return self.fill[index], self.line[index]

    def __iter__(self):
        """Iterate over (fill, line) color pairs."""
        return zip(self.fill, self.line)


# Type-safe overloads
@overload
def generate_color_palette(
    n_colors: int = 5,
    alpha: float = 1.0,
    *,
    format: Literal["matplotlib"] = "matplotlib",
) -> list[tuple[float, float, float, float]]: ...


@overload
def generate_color_palette(
    n_colors: int = 5,
    alpha: float = 1.0,
    *,
    format: Literal["plotly"],
) -> PlotlyColorPalette: ...

@overload
def generate_color_palette(
    n_colors: int = 5,
    alpha: float = 1.0,
    *,
    format: Literal["hex"],
) -> list[str]: ...


def generate_color_palette(
    n_colors: int = 5,
    alpha: float = 1.0,
    *,
    format: Literal["matplotlib", "plotly", "hex"] = "matplotlib",
):
    """Generate a harmonious color palette.

    Examples
    --------
    >>> # Matplotlib
    >>> colors = generate_color_palette(5, alpha=0.8, format="matplotlib")
    >>> ax.pie(counts, colors=colors)

    >>> # Plotly
    >>> palette = generate_color_palette(5, alpha=0.2, format="plotly")
    >>> for i, (fill, line) in enumerate(palette):
    ...     fig.add_trace(go.Scatterpolar(fillcolor=fill, line=dict(color=line)))

    >>> # Hex
    >>> colors = generate_color_palette(5, format="hex")
    >>> # ['#3498db', '#e74c3c', ...]
    """

    base_colors = [
        (52, 152, 219),
        (231, 76, 60),
        (46, 204, 113),
        (155, 89, 182),
        (241, 196, 15),
        (230, 126, 34),
        (26, 188, 156),
        (149, 165, 166),
        (243, 156, 18),
        (192, 57, 43),
    ]

    if n_colors > len(base_colors):
        raise ValueError(f"n_colors must be <= {len(base_colors)}")

    selected = base_colors[:n_colors]

    if format == "matplotlib":
        return [(r / 255, g / 255, b / 255, alpha) for r, g, b in selected]

    elif format == "plotly":
        fill_colors = [f"rgba{(*rgb, alpha)}" for rgb in selected]
        line_colors = [f"rgb{rgb}" for rgb in selected]
        return PlotlyColorPalette(fill=fill_colors, line=line_colors)

    elif format == "hex":
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in selected]

    raise ValueError(f"Unknown format: {format}")
