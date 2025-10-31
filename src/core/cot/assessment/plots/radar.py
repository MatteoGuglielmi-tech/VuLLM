from pathlib import Path

import plotly.graph_objects as go


def plot_judge_comparison_radar(judge_evaluations: dict[str, dict[str, float | int]], output_dir: Path):
    """Radar chart comparing judge criteria (THIS is where radar makes sense!)."""

    categories = list(next(iter(judge_evaluations.values())).keys())

    fig = go.Figure()

    colors = ["rgba(52, 152, 219, 0.2)", "rgba(231, 76, 60, 0.2)", "rgba(46, 204, 113, 0.2)"]
    line_colors = ["rgb(52, 152, 219)", "rgb(231, 76, 60)", "rgb(46, 204, 113)"]

    for idx, (judge_name, scores) in enumerate(judge_evaluations.items()):
        values = list(scores.values())

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=judge_name,
                fillcolor=colors[idx],
                line=dict(color=line_colors[idx], width=3),
                marker=dict(size=8, symbol="circle"),
            )
        )

    fig.update_layout(
        title=dict(
            text="<b>Judge Performance Across Evaluation Criteria</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Verdana, verdana", color="#2c3e50"),
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="<b>Judges</b>", font=dict(size=14)),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=13),
            bgcolor="rgba(229, 236, 246, 255)",
            bordercolor="#7f8c8d",
            borderwidth=2,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=True,
                linewidth=2,
                linecolor="#7f8c8d",
                gridcolor="rgba(127, 140, 141, 0.4)",
                gridwidth=2,
                tickfont=dict(size=12, color="#2c3e50"),
                tickmode="linear",
                tick0=0,
                dtick=0.2,
            ),
            angularaxis=dict(
                linewidth=2,
                linecolor="#7f8c8d",
                gridcolor="rgba(127, 140, 141, 0.5)",  # More visible
                gridwidth=2,
                tickfont=dict(size=13, family="Arial, sans-serif", color="#2c3e50"),
            ),
            # bgcolor='rgba(236, 240, 241, 0.3)'
        ),
        paper_bgcolor="white",
        width=900,
        height=700,
        margin=dict(l=80, r=180, t=120, b=80),
    )

    fig.write_image(
        output_dir / "radar_plot.png",
        width=1200,
        height=800,
        scale=2,  # Increases resolution (2x = higher quality)
    )
