import torch
from transformers.generation.logits_process import LogitsProcessor


class EnforceSingleTokenGeneration(LogitsProcessor):
    """
    A LogitsProcessor that forces the model to generate only one of the
    allowed tokens and then stop.
    """

    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Args:
            input_ids (torch.Tensor): The input IDs for the generation.
            scores (torch.Tensor): The logits for the next token.

        Returns:
            torch.Tensor: The modified logits.
        """

        _ = input_ids

        # Create a mask that sets all token probabilities to negative infinity
        mask: torch.Tensor = torch.full_like(input=scores, fill_value=-float("inf"))
        # Set the probability of our allowed tokens to 0 (leaving their original logits)
        mask[:, self.allowed_token_ids] = 0

        return scores + mask
