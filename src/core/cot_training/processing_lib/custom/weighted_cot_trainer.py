import logging
import torch

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer
from trl.trainer.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


class WeightedCoTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that applies token-level loss weighting.

    Model-agnostic: Uses tokenizer to detect role boundaries instead of hardcoded delimiters.
    """

    def __init__(
        self,
        *args,
        tokenizer: PreTrainedTokenizer,
        reasoning_weight: float = 1.0,
        answer_weight: float = 1.5,
        answer_marker: str = "Final Answer:",
        **kwargs,
    ):
        super().__init__(*args, processing_class=tokenizer, **kwargs)

        self.reasoning_weight = reasoning_weight
        self.answer_weight = answer_weight
        self.answer_marker = answer_marker

        assert self.tokenizer is not None
        self.answer_marker_tokens = self.tokenizer.encode(answer_marker, add_special_tokens=False)

        logger.info(f"🎯 Weighted Loss Configuration:")
        logger.info(f"   Reasoning weight: {reasoning_weight}")
        logger.info(f"   Answer weight: {answer_weight}")
        logger.info(f"   Answer marker: '{answer_marker}'")
        logger.info(f"   Answer marker tokens: {self.answer_marker_tokens}")

    def _get_assistant_token_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Model-agnostic way to identify assistant tokens.

        Method: Assistant tokens are those with labels != -100
        (This works because of assistant_only_loss=True in SFTConfig)

        Parameters
        ----------
        input_ids, torch.Tensor
            Tensor of tokenized ids of [seq_len] length
        labels, torch.Tensor 
            [seq_len]

        Returns:
            mask: [seq_len] boolean tensor (True = assistant token)
        """
        return labels != -100

    def _find_answer_start_position(self, input_ids: torch.Tensor) -> int|None:
        """Find the starting position of answer marker tokens in sequence.

        Parameters
        ----------
        input_ids: [seq_len]

        Returns
        -------
            Position of first token of answer marker, or None if not found
        """

        marker_len = len(self.answer_marker_tokens) # how many tokens
        if marker_len == 0: return None

        marker_tensor = torch.tensor(self.answer_marker_tokens, device=input_ids.device, dtype=input_ids.dtype)

        # sliding window search to avoid matching the wrong token
        for i in range(len(input_ids) - marker_len + 1):
            if torch.all(input_ids[i : i + marker_len] == marker_tensor):
                return i

        return None

    def _create_weight_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Create per-token loss weight mask.

        Strategy:
        1. Find assistant tokens (labels != -100)
        2. Within assistant tokens, find "Final Answer:" marker
        3. Apply reasoning_weight before marker, answer_weight after

        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len]

        Returns:
            weight_mask: [batch_size, seq_len]
        """
        batch_size, _ = input_ids.shape
        weight_mask = torch.ones_like(labels, dtype=torch.float32)

        for batch_idx in range(batch_size):
            # get assistant token mask
            assistant_mask = self._get_assistant_token_mask(labels[batch_idx])

            # find first assistant token position
            assistant_positions = torch.where(assistant_mask)[1]
            if len(assistant_positions) == 0: continue
            assistant_start = assistant_positions[0].item()
            assistant_end = assistant_positions[-1].item() + 1

            # find answer marker position within this sample
            answer_start = self._find_answer_start_position(input_ids[batch_idx])

            if answer_start is not None and answer_start >= assistant_start: # marker found
                # Reasoning: from assistant_start to answer_start
                weight_mask[batch_idx, assistant_start:answer_start] = self.reasoning_weight
                # Answer: from answer_start to end of assistant
                weight_mask[batch_idx, answer_start:assistant_end] = self.answer_weight
            else:
                # no marker found, treat all assistant tokens as reasoning
                weight_mask[batch_idx, assistant_start:assistant_end] = (self.reasoning_weight)

            # Ensure masked tokens (label == -100) have 0 weight to ignore them during loss computation
            weight_mask[batch_idx][labels[batch_idx] == -100] = 0.0

        return weight_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted loss."""

        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal language modeling
        #
        # why shifting?
        # # Without shifting:
        # Position:        0      1      2      3      4      5
        # input_ids:    [8468,  220,   16,    25,   4343,  ...]
        #                Step    _     1      :    Check
        #
        # logits[pos]:  [pred   pred   pred   pred   pred   pred ]
        #                for1   for2   for3   for4   for5   for6
        #
        # labels:       [8468,  220,   16,    25,   4343,  ...]
        #                Step    _     1      :    Check
        #
        # Problem: logits[0] predicts position 1, but labels[0] is position 0!
        # Comparing the prediction for position 1 with the label for position 0!

        shift_logits = logits[..., :-1, :].contiguous() # drop last prediction since we reached the max sequence length
        shift_labels = labels[..., 1:].contiguous() # drop first target prediction since at posizione 0, model has no context
        shift_input_ids = input_ids[..., :-1].contiguous() # align with shift_labels

        weight_mask = self._create_weight_mask(shift_input_ids, shift_labels)

        # compute per-token loss
        loss_fct = CrossEntropyLoss(reduction="none") # tensor with one loss per token position
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Apply weights
        weighted_loss = loss * weight_mask.view(-1)

        # Average over non-masked tokens, accounting for weights
        mask = (shift_labels.view(-1) != -100).float()

        # Weighted average: sum(weighted_loss) / sum(weights)
        total_weighted_loss = (weighted_loss * mask).sum()
        total_weight = (weight_mask.view(-1) * mask).sum()

        final_loss = (
            total_weighted_loss / total_weight
            if total_weight > 0
            else total_weighted_loss
        )

        return (final_loss, outputs) if return_outputs else final_loss



# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.trainer.sft_config import SFTConfig

    # Your setup
    tokenizer = AutoTokenizer.from_pretrained("your_model")
    model = AutoModelForCausalLM.from_pretrained("your_model")
    dataset = load_from_disk("your_dataset")

    # Visualize how weights will be applied (debug before training!)
    sample = {
        "system_prompt": "You are an expert cybersecurity analyst...",
        "user_prompt": "Analyze this C code: ...",
        "assistant_content": "Step 1: ... Step 2: ... Final Answer: YES (CWE-119)",
    }

    # Create trainer (model-agnostic!)
    training_args = SFTConfig(
        output_dir="./weighted_training",
        # ... your args ...
    )

    trainer = WeightedCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        reasoning_weight=1.0,
        answer_weight=1.5,
        answer_marker="Final Answer:",  # Only this is task-specific!
    )

    trainer.train()
