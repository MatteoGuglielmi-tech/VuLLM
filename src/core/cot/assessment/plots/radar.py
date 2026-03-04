import plotly.graph_objects as go

from pathlib import Path

from .colors import generate_color_palette


def plot_judge_comparison_radar(judge_evaluations: dict[str, dict[str, float | int]], output_dir: Path):
    """Radar chart comparing judge criteria."""

    categories = list(next(iter(judge_evaluations.values())).keys())

    fig = go.Figure()

    colors, line_colors = generate_color_palette(5, alpha=0.2)
    val_for_legend = {}

    for idx, (judge_name, scores) in enumerate(judge_evaluations.items()):
        values = list(scores.values())
        val_for_legend[judge_name] = values

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

    fig.add_annotation(
        x=2,
        y=1,
        text=(
            f"Qwen2.5-Coder-32B =  { ",".join([x.astype(str) for x in val_for_legend["Qwen2.5-Coder-32B-Instruct-bnb-4bit"]])}",
            f"Qwen2.5-72B =  { ",".join([x.astype(str) for x in val_for_legend["Qwen2.5-72B"]])}",
            f"Phi-4 =  { ",".join([x.astype(str) for x in val_for_legend["Phi-4"]])}",
            f"DeepSeek-R1-Distill-Llama =  {",".join([x.astype(str) for x in val_for_legend["DeepSeek-R1-Distill-Llama"]]) }",
        ),
    )

    fig.write_image(
        output_dir / "radar_plot.png",
        width=1200,
        height=800,
        scale=2,  # Increases resolution (2x = higher quality)
    )
