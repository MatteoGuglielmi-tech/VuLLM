import plotly.graph_objects as go

from pathlib import Path


def plot_filtering_sankey(stats, output_dir: Path):
    """Sankey diagram showing filtering flow."""

    low_qual_only = stats["low_quality"] - stats["both_issues"]
    low_agree_only = stats["low_agreement"] - stats["both_issues"]
    both = stats["both_issues"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[
                        f"Total Samples<br>({stats['total']})",
                        f"Kept<br>({stats['kept']})",
                        f"Rejected<br>({stats['rejected']})",
                        f"Low Quality<br>({low_qual_only})",
                        f"Low Agreement<br>({low_agree_only})",
                        f"Both Issues<br>({both})",
                    ],
                    color=[
                        "lightblue",
                        "lightgreen",
                        "salmon",
                        "#ff6b6b",
                        "#ffa07a",
                        "#cd5c5c",
                    ],
                ),
                link=dict(
                    source=[0, 0, 2, 2, 2],  # indices of source nodes
                    target=[1, 2, 3, 4, 5],  # indices of target nodes
                    value=[
                        stats["kept"],
                        stats["rejected"],
                        low_qual_only,
                        low_agree_only,
                        both,
                    ],
                    color=[
                        "rgba(144, 238, 144, 0.4)",
                        "rgba(250, 128, 114, 0.4)",
                        "rgba(255, 107, 107, 0.4)",
                        "rgba(255, 160, 122, 0.4)",
                        "rgba(205, 92, 92, 0.4)",
                    ],
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Filtering Pipeline Flow (Keep Rate: {stats['keep_rate']:.1%})</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Verdana, verdana", color="#2c3e50"),
        ),
        paper_bgcolor="white",
        width=900,
        height=700,
        margin=dict(l=80, r=180, t=120, b=80),
    )

    fig.write_image(
        output_dir / "sankey.png",
        width=1200,
        height=800,
        scale=2,
    )
