import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import cast
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

from .ui import rich_exception, stateless_progress, rich_status


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CWEInfo:
    status: str
    name: str
    description: str = ""


class CWEStatusAnalyzer:

    def __init__(
        self,
        dataset_path: Path | str,
        cwe_status_csv: Path | str,
        output_dir: Path | str,
    ):
        """
        Initialize CWE Status Analyzer.

        Parameters
        ----------
        dataset_path : Path | str
            Path to JSONL dataset
        cwe_status_csv : Path | str
            Path to MITRE CWE status CSV
        output_dir : Path | str
            Output directory for results
        """

        self.dataset_path = Path(dataset_path)
        self.cwe_status_csv = Path(cwe_status_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            if not self.cwe_status_csv.exists():
                raise FileNotFoundError(f"CWE status CSV not found: {self.cwe_status_csv}")
        except FileNotFoundError:
            rich_exception()

        with rich_status(f"📂 Loading dataset: {self.dataset_path.name}"):
            self.dataset_df = pd.read_json(str(self.dataset_path), lines=True)

        with rich_status(f"📂 Loading CWE status: {self.cwe_status_csv.name}"):
            self.cwe_status_df = pd.read_csv(str(self.cwe_status_csv), index_col=False)

        logger.info(f"Loaded {len(self.dataset_df)} records")
        logger.info(f"Loaded {len(self.cwe_status_df)} CWE definitions")

        self._prepare_cwe_mappings()

    def _prepare_cwe_mappings(self):
        """Prepare CWE ID and status mappings."""

        self.mitre_cwes = set(
            f"CWE-{cwe_id}" for cwe_id in self.cwe_status_df["CWE-ID"].astype(str)
        )

        self.cwe_status_lookup = {
            f"CWE-{row['CWE-ID']}": CWEInfo(
                status=cast(str, row["Status"]),
                name=cast(str, row["Name"]),
                description=cast(str, row.get("Description", "")),
            )
            for _, row in self.cwe_status_df.iterrows()
        }


    def analyze_cwe_status(self) -> dict:
        """Comprehensive CWE status analysis."""

        all_cwes: list[str] = []
        for cwe_list_str in self.dataset_df["cwe"]:
            if isinstance(cwe_list_str, str):
                cwes: list[str] = (
                    eval(cwe_list_str)
                    if cwe_list_str.startswith("[")
                    else [cwe_list_str]
                )
                all_cwes.extend(cwes)
            elif isinstance(cwe_list_str, list):
                all_cwes.extend(cwe_list_str)

        cwe_counts = Counter(all_cwes)
        unique_cwes: set[str] = set(all_cwes)

        deprecated_cwes: set[str] = set()
        draft_cwes: set[str] = set()
        stable_cwes: set[str] = set()
        unknown_cwes: set[str] = set()

        for cwe in unique_cwes:
            if cwe not in self.mitre_cwes:
                unknown_cwes.add(cwe)
            else:
                cwe_info: CWEInfo = self.cwe_status_lookup[cwe]
                status = cwe_info.status
                if status == "Deprecated":
                    deprecated_cwes.add(cwe)
                elif status == "Draft":
                    draft_cwes.add(cwe)
                elif status in ["Stable", "Incomplete"]:
                    stable_cwes.add(cwe)

        # Count functions affected
        deprecated_func_count: int = sum(
            1
            for cwe_list_str in self.dataset_df["cwe"]
            if any(
                cwe in deprecated_cwes
                for cwe in (
                    eval(cwe_list_str)
                    if isinstance(cwe_list_str, str)
                    else cwe_list_str
                )
            )
        )
        unknown_func_count: int = sum(
            1
            for cwe_list_str in self.dataset_df["cwe"]
            if any(
                cwe in unknown_cwes
                for cwe in (
                    eval(cwe_list_str)
                    if isinstance(cwe_list_str, str)
                    else cwe_list_str
                )
            )
        )

        # Calculate occurrences by category
        deprecated_occurrences = sum(cwe_counts[cwe] for cwe in deprecated_cwes)
        unknown_occurrences = sum(cwe_counts[cwe] for cwe in unknown_cwes)

        return {
            "total_unique_cwes": len(unique_cwes),
            "total_cwe_occurrences": sum(cwe_counts.values()),
            "deprecated": {
                "count": len(deprecated_cwes),
                "cwes": deprecated_cwes,
                "occurrences": deprecated_occurrences,
                "func_count": deprecated_func_count,
            },
            "draft": {
                "count": len(draft_cwes),
                "cwes": draft_cwes,
            },
            "stable": {
                "count": len(stable_cwes),
                "cwes": stable_cwes,
            },
            "unknown": {
                "count": len(unknown_cwes),
                "cwes": unknown_cwes,
                "occurrences": unknown_occurrences,
                "func_count": unknown_func_count,
            },
            "cwe_counts": cwe_counts,
        }

    def plot_cwe_status_overview(self, stats: dict):
        """Bar chart showing CWE status distribution."""

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Unique CWE count by status
        categories = ["Stable", "Draft", "Deprecated", "Unknown"]
        counts = [
            stats["stable"]["count"],
            stats["draft"]["count"],
            stats["deprecated"]["count"],
            stats["unknown"]["count"],
        ]
        pal = sns.color_palette(palette="pastel", n_colors=len(categories))

        bars1 = ax1.bar(
            categories, counts, color=pal, edgecolor="black", linewidth=1.5
        )
        ax1.set_ylabel("Number of Unique CWEs", fontsize=12, fontweight="bold")
        ax1.set_title(
            "CWE Status Distribution (Unique CWEs)", fontsize=14, fontweight="bold"
        )
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Right: Function count affected
        func_categories = ["With Deprecated CWEs", "With Unknown CWEs"]
        func_counts = [
            stats["deprecated"]["func_count"],
            stats["unknown"]["func_count"],
        ]
        func_colors = sns.color_palette(palette="pastel", n_colors=2)
        bars2 = ax2.bar(
            func_categories,
            func_counts,
            color=func_colors,
            edgecolor="black",
            linewidth=1.5,
        )
        ax2.set_ylabel("Number of Functions", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Functions with Problematic CWE Labels", fontsize=14, fontweight="bold"
        )
        ax2.grid(axis="y", alpha=0.3)

        for bar, count in zip(bars2, func_counts):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        save_path = self.output_dir / "cwe_status_overview.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_stacked_bar_deprecated_unknown(self, stats: dict):
        """Stacked bar showing composition of deprecated and unknown CWEs."""

        deprecated_cwes = stats["deprecated"]["cwes"]
        unknown_cwes = stats["unknown"]["cwes"]
        cwe_counts = stats["cwe_counts"]

        deprecated_sorted = sorted(
            [(cwe, cwe_counts[cwe]) for cwe in deprecated_cwes],
            key=lambda x: x[1],
            reverse=True,
        )

        unknown_sorted = sorted(
            [(cwe, cwe_counts[cwe]) for cwe in unknown_cwes],
            key=lambda x: x[1],
            reverse=True,
        )

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Generate distinct colors
        def get_colors(n):
            return plt.cm.tab20(np.linspace(0, 1, n))

        # Deprecated CWEs
        if deprecated_sorted:
            cwes_dep, counts_dep = zip(*deprecated_sorted)
            colors_dep = get_colors(len(cwes_dep))

            # Calculate percentages for stacking
            total_dep = sum(counts_dep)
            bottom = 0

            for i, (cwe, count) in enumerate(zip(cwes_dep, counts_dep)):
                pct = (count / total_dep) * 100
                ax1.barh(
                    "Deprecated",
                    pct,
                    left=bottom,
                    color=colors_dep[i],
                    edgecolor="white",
                    linewidth=2,
                    label=f"{cwe} ({count})",
                )

                # Add text if segment is large enough
                if pct > 5:
                    ax1.text(
                        bottom + pct / 2,
                        0,
                        f"{cwe}\n{count}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )

                bottom += pct

            ax1.set_xlim(0, 100)
            ax1.set_xlabel("Percentage", fontsize=12, fontweight="bold")
            ax1.set_title(
                f"Deprecated CWE Composition (Total: {total_dep})",
                fontsize=14,
                fontweight="bold",
            )
            ax1.legend(
                loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, frameon=False
            )

        # Unknown CWEs
        if unknown_sorted:
            cwes_unk, counts_unk = zip(*unknown_sorted)
            colors_unk = get_colors(len(cwes_unk))

            total_unk = sum(counts_unk)
            bottom = 0

            for i, (cwe, count) in enumerate(zip(cwes_unk, counts_unk)):
                pct = (count / total_unk) * 100
                ax2.barh(
                    "Unknown",
                    pct,
                    left=bottom,
                    color=colors_unk[i],
                    edgecolor="white",
                    linewidth=2,
                    label=f"{cwe} ({count})",
                )

                if pct > 5:
                    ax2.text(
                        bottom + pct / 2,
                        0,
                        f"{cwe}\n{count}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )

                bottom += pct

            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Percentage", fontsize=12, fontweight="bold")
            ax2.set_title(
                f"Unknown CWE Composition (Total: {total_unk})",
                fontsize=14,
                fontweight="bold",
            )
            ax2.legend(
                loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, frameon=False
            )

        plt.tight_layout()
        save_path = self.output_dir / "cwe_composition_stacked.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def generate_report(self, stats: dict):
        """Generate markdown report."""

        report: list[str] = []
        report.append("# CWE Status Analysis Report\n")
        report.append(f"**Dataset**: {self.dataset_path.name}\n")
        report.append(f"**Total Functions**: {len(self.dataset_df)}\n")
        report.append(f"**Total Unique CWEs**: {stats['total_unique_cwes']}\n")
        report.append(
            f"**Total CWE Occurrences**: {stats['total_cwe_occurrences']}\n\n"
        )

        report.append("## Status Breakdown\n\n")

        report.append("### Stable CWEs\n")
        report.append(f"- **Count**: {stats['stable']['count']}\n")
        report.append(f"- **CWEs**: {', '.join(sorted(stats['stable']['cwes']))}\n\n")

        report.append("### Draft CWEs\n")
        report.append(f"- **Count**: {stats['draft']['count']}\n")
        report.append(f"- **CWEs**: {', '.join(sorted(stats['draft']['cwes']))}\n\n")

        report.append("### ⚠️ Deprecated CWEs\n")
        report.append(f"- **Count**: {stats['deprecated']['count']}\n")
        report.append(
            f"- **Occurrences in Dataset**: {stats['deprecated']['occurrences']}\n"
        )
        report.append(
            f"- **Functions Affected**: {stats['deprecated']['func_count']}\n"
        )
        report.append(
            f"- **CWEs**: {', '.join(sorted(stats['deprecated']['cwes']))}\n\n"
        )

        # Top deprecated by occurrence
        if stats["deprecated"]["cwes"]:
            deprecated_by_count = sorted(
                [
                    (cwe, stats["cwe_counts"][cwe])
                    for cwe in stats["deprecated"]["cwes"]
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            report.append("**Top 5 Deprecated CWEs by Occurrence**:\n")
            for cwe, count in deprecated_by_count:
                report.append(f"  - {cwe}: {count} occurrences\n")
            report.append("\n")

        report.append("### ❌ Unknown CWEs (Not in MITRE)\n")
        report.append(f"- **Count**: {stats['unknown']['count']}\n")
        report.append(
            f"- **Occurrences in Dataset**: {stats['unknown']['occurrences']}\n"
        )
        report.append(f"- **Functions Affected**: {stats['unknown']['func_count']}\n")
        report.append(f"- **CWEs**: {', '.join(sorted(stats['unknown']['cwes']))}\n\n")

        # Top unknown by occurrence
        if stats["unknown"]["cwes"]:
            unknown_by_count = sorted(
                [(cwe, stats["cwe_counts"][cwe]) for cwe in stats["unknown"]["cwes"]],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            report.append("**Top 5 Unknown CWEs by Occurrence**:\n")
            for cwe, count in unknown_by_count:
                report.append(f"  - {cwe}: {count} occurrences\n")

        # Save report
        report_path = self.output_dir / "cwe_status_report.md"
        with open(report_path, "w") as f:
            f.writelines(report)

    def run_analysis(self):
        """Run complete CWE status analysis."""

        with rich_status(description="🔍 Analyzing CWE status "):
            stats = self.analyze_cwe_status()

        with stateless_progress("📊 Generating visualizations") as status:
            status.update("Creating status overview...")
            self.plot_cwe_status_overview(stats)
            status.update("Creating problematic analysis barplot...")
            self.plot_stacked_bar_deprecated_unknown(stats)
            status.stop()

        with stateless_progress("📝 Generating report ") as status:
            self.generate_report(stats)
            status.stop()

        logger.info("[bold green]✓[/bold green] Analysis complete!")

        return stats
