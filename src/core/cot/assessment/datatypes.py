from dataclasses import dataclass

import numpy as np


@dataclass
class ReasoningSample:
    """Container for reasoning samples"""

    project: str
    cwe: list[str]
    target: int
    func: str
    cwe_desc: list[str]
    reasoning: str
    sample_id: str | None = None


@dataclass
class EvaluationResult:
    """Container for detailed evaluation results"""

    judge_name: str
    quality_score: float
    correctness: float
    completeness: float
    clarity: float
    technical_accuracy: float
    logical_flow: float
    confidence: float
    justification: str
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "judge_name": self.judge_name,
            "quality_score": self.quality_score,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "technical_accuracy": self.technical_accuracy,
            "logical_flow": self.logical_flow,
            "confidence": self.confidence,
            "justification": self.justification,
            "parse_error": self.parse_error,
        }

    @staticmethod
    def get_criteria_names() -> list[str]:
        """Get canonical order of criteria"""
        return [
            "correctness",
            "completeness",
            "clarity",
            "technical_accuracy",
            "logical_flow",
        ]

    def get_criteria_vector(self) -> np.ndarray:
        """Get all criterion scores as a vector"""
        return np.array([getattr(self, name) for name in self.get_criteria_names()])
