def generate_color_palette(
    n_colors: int = 5, alpha: float = 0.2
) -> tuple[list[str], list[str]]:
    """Generate a harmonious color palette.

    Parameters
    ----------
    n_colors : int
        Number of colors to generate
    alpha : float
        Opacity for filled colors (0.0 to 1.0)

    Returns
    -------
    tuple[list[str], list[str]]
        (fill_colors, line_colors)
    """

    base_colors = [
        (52, 152, 219),  # Blue
        (231, 76, 60),  # Red
        (46, 204, 113),  # Green
        (155, 89, 182),  # Purple
        (241, 196, 15),  # Yellow
        (230, 126, 34),  # Orange
        (26, 188, 156),  # Teal
        (149, 165, 166),  # Gray
        (243, 156, 18),  # Dark Yellow
        (192, 57, 43),  # Dark Red
    ]

    selected = base_colors[:n_colors]

    # Generate rgba and rgb strings
    fill_colors = [f"rgba{(*rgb, alpha)}" for rgb in selected]
    line_colors = [f"rgb{rgb}" for rgb in selected]

    return fill_colors, line_colors
