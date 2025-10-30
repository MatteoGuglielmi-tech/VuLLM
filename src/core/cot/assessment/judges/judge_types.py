from dataclasses import dataclass


@dataclass
class JudgeConfig:
    """Configuration for each judge model"""

    model_name: str
    chat_template: str | None = None
    max_seq_length: int = 8192  # power of 2 and multiple of 512 + covers all prompts with margin
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_k: int = 50
    top_p: float = 1.
    min_p: float = 0.
    repetition_penalty: float = 1.05

    specialization: str = "general"
    description: str | None = None
