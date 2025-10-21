import logging
import os
import re
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from collections import Counter
from dataclasses import dataclass
from matplotlib import pyplot as plt
# from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from openai.types.chat import ChatCompletionMessageParam

from src.core.cot.assessment import utils
from src.core.cot.generation.llm_clients.gpt import AzureCoTGenerator

logger = logging.getLogger(__name__)


@dataclass
class QualityAssessor:
    """Performs a closed-loop quality assessment on a sample of generated reasonings using an LLM as a judge."""

    deployment_name: str
    max_completion_tokens: int
    client: AzureCoTGenerator

    JUDGE_SYSTEM_PROMPT = (
        "You are a meticulous cybersecurity analyst and **LLM trainer**. "
        "Your task is to evaluate an AI-generated reasoning chains to determine whether they are high-quality enough for a fine-tuning dataset.\n"
        "You must be critical, objective, and strictly follow the provided rubric."
    ).strip()

    JUDGE_USER_PROMPT_TEMPLATE = (
        "Please evaluate the following reasoning chain for a code vulnerability analysis.\n"
        "The C code to analyze is provided in the `C Code to Analyze` section below."
        "**Evaluation Rubric:**\n"

        "1. **Correctness**: Is the analysis factually correct with respect to the provided C code? Does it correctly identify the vulnerability (or lack thereof) as well as leading to the final conclusion? This is the most important criterion.\n"
        "2. **Coherence**: Is the reasoning logical, step-by-step and easy for a human to follow? Additionally, is it sufficiently qualitatively good to be used to drive the fine-tuning of a specialized LLM agent for vulnerability detection and classification? It must be a clear, coherent thought process.\n"
        "3. **Format Adherence**: Does the response end with a correctly formatted `Final Answer:` line (e.g., `Final Answer: YES (CWE-XXX)` or `Final Answer: NO`)?\n\n"

        "Based on this rubric, provide a final score from 1 (unusable) to 5 (perfect) and a brief, actionable suggestion for improvement if the score is below 5.\n\n"

        "**Output Format:**\n"
        "The output must adhere to the following two line format:\n"
        "Score: [1-5]\n"
        "Reason: [A single sentence briefly justifying the score.]\n"
        "Suggestions: [A list of improvements for each problematic point]\n\n"

        "**Output Format Example:**\n"
        "Score: 3\n"
        "Reason: The analysis correctly identifies the risky function but fails to explain the exact mechanism of the overflow. Additionally, it assigns CWE-94 without direct evicences in the analysed code.\n"
        "Suggestions:\n"
        "  - Step 3 should explicitly state that `strcpy` does not perform bounds checking.\n"
        "  - Step 4 incorrectly attributes CWE-94 (code injection) without evidence that the copied data is ever executed as code; it should focus on memory corruption (e.g., CWE-119) and clarify that the vulnerability depends on how `dst` is managed, not just the copy operation itself."

        "If the reasoning is perfect, the suggestion should be 'None'.\n\n"

        "--- C CODE TO ANALYZE ---\n"
        "{c_code}\n\n"

        "--- REASONING TO EVALUATE ---\n"
        "{reasoning}"
    ).strip()

    def _get_quality_score(self, reasoning: str, c_code: str) -> dict:
        """Uses the LLM judge to score a single reasoning chain."""

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self.JUDGE_USER_PROMPT_TEMPLATE.format(reasoning=reasoning, c_code=c_code)},
        ]

        try:
            response = self.client.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                temperature=0.0, # no creativity here just factual
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            evaluation_text = response.choices[0].message.content or ""
            score_match = re.search(r"Score:\s*(\d)", evaluation_text)
            reason_match = re.search(r"Reason:\s*(.*)", evaluation_text)
            suggestions_match = re.search(r"Suggestions:\s*(.*)", evaluation_text, re.DOTALL)

            score = int(score_match.group(1)) if score_match else 0
            reason = reason_match.group(1).strip() if reason_match else "Parsing failed."
            suggestions = suggestions_match.group(1).strip() if suggestions_match else "Parsing failed."
            return {"score": score, "reason": reason, "suggestions": suggestions}

        except Exception as e:
            logger.error(f"API call failed during evaluation: {e}")
            return {"score": 0, "reason": "API Error", "suggestions": "API Error"}

    def generate_report(self, assessed_results: list, output_dir: str):
        scores = [ res["quality_assessment"]["score"] for res in assessed_results if "quality_assessment" in res ]
        avg_score = sum(scores) / len(scores) if scores else 0
        score_distribution = Counter(scores)

        output_fp = os.path.join(output_dir, "report.txt")

        with open(file=output_fp, mode="w", encoding="utf-8") as f:
            f.write("=" * 20 + " Reasoning Quality Assessment Report " + "=" * 20 + "\n\n")
            f.write(f"Total Samples Assessed: {len(scores)}\n")
            f.write(f"Average Score: {avg_score:.2f} / 5.0\n")
            f.write("Score Distribution:\n")
            for score, count in sorted(score_distribution.items()):
                f.write(f"  - Score {score}: {count} samples\n")

            f.write("\n\n" + "=" * 20 + " Detailed Review for Low-Scoring Samples (Score < 4) " + "=" * 20 + "\n\n")

            low_scoring_samples = [
                res
                for res in assessed_results
                if res.get("quality_assessment", {}).get("score", 5) < 4
            ]
            for sample in low_scoring_samples:
                f.write(f"--- Sample (Target: {sample['target']}, CWEs: {sample['cwe']}) ---\n")
                f.write(f"Quality Score: {sample['quality_assessment']['score']}\n")
                f.write(f"Reason: {sample['quality_assessment']['reason']}\n")
                f.write(f"Suggestions:\n{sample['quality_assessment']['suggestions']}\n")
                f.write("Original Reasoning:\n" + sample["reasoning"] + "\n\n")

        logger.info(f"✅ Quality assessment report saved to: {output_fp}")


    def _plot_score_distribution(self, assessed_results: list, output_dir: str):
        """Creates and saves a bar chart of the score distribution."""

        scores = [res["quality_assessment"]["score"] for res in assessed_results if "quality_assessment" in res]
        if not scores:
            logger.warning("No scores found to plot.")
            return

        score_counts = Counter(scores)

        df_scores = pd.DataFrame({
            "Score": range(1, 6),
            "Count": [score_counts.get(i, 0) for i in range(1, 6)]
        })

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = sns.color_palette(palette="pastel", n_colors=5)

        barplot = sns.barplot(
            x="Score",
            y="Count",
            data=df_scores,
            hue="Score",
            palette=colors,
            ax=ax,
            edgecolor="black",
            linewidth=1.5,
            dodge=False,
            legend=False
        )

        for index, row in df_scores.iterrows():
            if row.Count > 0:
                barplot.text(
                    index, # type: ignore
                    row.Count + (df_scores["Count"].max() * 0.02),
                    f"{int(row.Count)}",
                    color="black",
                    ha="center",
                    fontsize=12
                )

        sns.despine()

        # legend_handles = [Patch(facecolor=colors[i], edgecolor='black', label=f'Score {i+1}') for i in range(5)]
        # legend_labels = [f"Score {i+1} [{df_scores.loc[i, 'Count']}]" for i in range(5)]

        # ax.legend(
        #     handles=legend_handles,
        #     labels=legend_labels,
        #     title="Score Counts",
        #     loc="best",
        #     facecolor="white",
        #     fontsize=12,
        #     title_fontsize=14,
        #     bbox_to_anchor=[0.85, 0.9],
        #     bbox_transform=fig.transFigure,
        # )

        ax.tick_params(axis="x", labelrotation=0)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_formatter("{x:.0f}")

        ax.grid(True, which="minor", linestyle=":", alpha=0.6, axis="y", color="darkgray")
        ax.grid(True, which="major", linestyle="-", alpha=0.8, axis="y", color="darkgray")

        plt.title("Distribution of Reasoning Quality Scores", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Quality Score (1=Poor, 5=Excellent)", fontsize=14, labelpad=15)
        plt.ylabel("Number of Samples", fontsize=14, labelpad=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "score_distribution.png")
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"✅ Score distribution plot saved to: {plot_path}")

    def run_assessment(self, input_fp: str, output_dir: str, sample_size: int):
        """Orchestrates the quality assessment by streaming a random sample."""

        logger.info(f"Streaming a random set of {sample_size} samples from {input_fp}")

        true_size, sample_pool = utils._stream_random_sample(input_fp=input_fp, sample_size=sample_size)

        assessed_results = []
        for entry in tqdm(iterable=sample_pool, total=true_size, desc="Assessing Reasoning Quality"):
            reasoning = entry.get("reasoning", "")
            c_code = entry.get("func", "")
            if not reasoning or not c_code: continue
            evaluation = self._get_quality_score(reasoning=reasoning, c_code=c_code)
            assessed_results.append({**entry, "quality_assessment": evaluation})

        os.makedirs(name=output_dir, exist_ok=True)
        self.generate_report(assessed_results=assessed_results, output_dir=output_dir)
        self._plot_score_distribution(assessed_results=assessed_results, output_dir=output_dir)



