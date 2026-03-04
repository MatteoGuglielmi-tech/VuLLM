"""
Advanced implementation of _get_assistant_token_mask with:
- Special token filtering
- Data preparation sanity checks
- Optional answer marker exclusion
"""

import torch
import logging

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer
from trl.trainer.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


# ============================================================================
# INTEGRATION WITH WeightedCoTTrainer
# ============================================================================
class WeightedCoTTrainerAdvanced(SFTTrainer):
    def __init__(
        self,
        *args,
        tokenizer: PreTrainedTokenizer,
        reasoning_weight: float = 1.0,
        answer_weight: float = 1.5,
        answer_marker: str = "Final Answer:",
        exclude_answer_marker_from_loss: bool = False,
        enable_mask_sanity_checks: bool = True,
        **kwargs,
    ):
        super().__init__(*args, processing_class=tokenizer, **kwargs)

        self.tokenizer = tokenizer
        self.reasoning_weight = reasoning_weight
        self.answer_weight = answer_weight
        self.answer_marker = answer_marker
        self.exclude_answer_marker_from_loss = exclude_answer_marker_from_loss
        self.enable_mask_sanity_checks = enable_mask_sanity_checks

        # Precompute answer marker tokens
        self.answer_marker_tokens = self.tokenizer.encode(
            answer_marker, add_special_tokens=False
        )

        logger.info("=" * 80)
        logger.info("🎯 Weighted CoT Trainer Configuration")
        logger.info("=" * 80)
        logger.info(f"Reasoning weight: {reasoning_weight}")
        logger.info(f"Answer weight: {answer_weight}")
        logger.info(f"Answer marker: '{answer_marker}'")
        logger.info(f"Marker token IDs: {self.answer_marker_tokens}")
        logger.info(f"Exclude marker from loss: {exclude_answer_marker_from_loss}")
        logger.info(f"Sanity checks enabled: {enable_mask_sanity_checks}")
        logger.info("=" * 80)

    def _get_assistant_token_mask_advanced(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        exclude_answer_marker: bool = False,
        enable_sanity_checks: bool = True,
    ) -> torch.Tensor:
        """
        Advanced assistant token identification with filtering and validation.

        Features:
        1. Filters out special tokens (EOS, BOS, PAD) from loss computation
        2. Optional: Exclude "Final Answer:" marker from loss
        3. Sanity checks to catch data preparation bugs

        Args:
            input_ids: [seq_len] Token IDs for the sequence
            labels: [seq_len] Labels where:
                    - labels[i] = -100 means "don't compute loss" (masked)
                    - labels[i] = token_id means "compute loss" (not masked)
            exclude_answer_marker: If True, exclude "Final Answer:" from loss.
                                   Default False - marker is useful for parsing.
            enable_sanity_checks: If True, perform validation checks.
                                 Set False in production for speed.

        Returns:
            mask: [seq_len] Boolean tensor where True = compute loss on this token

        Raises:
            AssertionError: If sanity checks fail (only when enable_sanity_checks=True)

        Example:
            >>> input_ids = torch.tensor([128000, ..., 8468, 220, 16, ..., 128009])
            >>> labels = torch.tensor([-100, ..., 8468, 220, 16, ..., 128009])
            >>> mask = self._get_assistant_token_mask_advanced(input_ids, labels)
            >>> # mask excludes system/user/special tokens, includes content
        """
        # Validate inputs
        assert (
            input_ids.shape == labels.shape
        ), f"Shape mismatch: input_ids {input_ids.shape} vs labels {labels.shape}"

        seq_len = len(input_ids)

        # Step 1: Basic mask from labels (assistant tokens)
        mask = labels != -100

        # Step 2: Sanity checks (catch data preparation bugs)
        if enable_sanity_checks:
            self._validate_labels_alignment(input_ids, labels, mask)

        # Step 3: Filter out special tokens
        mask = self._filter_special_tokens(input_ids, mask)

        # Step 4: Optionally exclude "Final Answer:" marker
        if exclude_answer_marker:
            mask = self._exclude_answer_marker(input_ids, mask)

        # Step 5: Final validation
        if enable_sanity_checks:
            self._validate_mask_sanity(mask, seq_len)

        return mask

    def _validate_labels_alignment(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """
        Validate that labels match input_ids where not masked.

        This catches data preparation bugs where:
        - Labels are shifted incorrectly
        - Masking is applied to wrong positions
        - Tokenization is inconsistent

        Args:
            input_ids: Token IDs
            labels: Label tensor
            mask: Boolean mask (True = not masked)

        Raises:
            AssertionError: If labels don't match input_ids at non-masked positions
        """
        # Check alignment for non-masked positions
        mismatches = []
        for i in range(len(input_ids)):
            if mask[i]:  # If this position should have loss computed
                if labels[i] != input_ids[i]:
                    mismatches.append(
                        {
                            "position": i,
                            "input_id": input_ids[i].item(),
                            "label": labels[i].item(),
                            "input_token": self.tokenizer.decode([input_ids[i]]),
                            "label_token": (
                                self.tokenizer.decode([labels[i]])
                                if labels[i] >= 0
                                else "N/A"
                            ),
                        }
                    )

        if mismatches:
            # Log first few mismatches
            logger.error("=" * 80)
            logger.error("DATA PREPARATION BUG DETECTED: Labels don't match input_ids")
            logger.error("=" * 80)
            for idx, mismatch in enumerate(mismatches[:5]):  # Show first 5
                logger.error(f"Mismatch {idx + 1}:")
                logger.error(f"  Position: {mismatch['position']}")
                logger.error(
                    f"  input_ids[{mismatch['position']}] = {mismatch['input_id']} ('{mismatch['input_token']}')"
                )
                logger.error(
                    f"  labels[{mismatch['position']}] = {mismatch['label']} ('{mismatch['label_token']}')"
                )

            if len(mismatches) > 5:
                logger.error(f"... and {len(mismatches) - 5} more mismatches")

            logger.error("=" * 80)
            logger.error("This indicates a bug in your data preparation pipeline!")
            logger.error("Check your dataset formatting and tokenization logic.")
            logger.error("=" * 80)

            raise AssertionError(
                f"Found {len(mismatches)} label mismatches. "
                f"See logs above for details. "
                f"This suggests a data preparation bug."
            )

    def _filter_special_tokens(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filter out special tokens from loss computation.

        Special tokens (EOS, BOS, PAD) are structural and always present.
        Training on them wastes capacity and can cause hallucinations.

        Args:
            input_ids: Token IDs
            mask: Current boolean mask

        Returns:
            Updated mask with special tokens excluded
        """
        # Get special token IDs from tokenizer
        special_token_ids = []

        # End-of-sequence token (e.g., <|eot_id|> = 128009 for Llama-3)
        if (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            special_token_ids.append(self.tokenizer.eos_token_id)

        # Beginning-of-sequence token (e.g., <|begin_of_text|> = 128000)
        if (
            hasattr(self.tokenizer, "bos_token_id")
            and self.tokenizer.bos_token_id is not None
        ):
            special_token_ids.append(self.tokenizer.bos_token_id)

        # Padding token (usually not in assistant content, but filter for safety)
        if (
            hasattr(self.tokenizer, "pad_token_id")
            and self.tokenizer.pad_token_id is not None
        ):
            special_token_ids.append(self.tokenizer.pad_token_id)

        # Filter out special tokens
        for special_token_id in special_token_ids:
            special_token_positions = input_ids == special_token_id
            mask = mask & ~special_token_positions

        return mask

    def _exclude_answer_marker(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exclude "Final Answer:" marker tokens from loss computation.

        The marker is a template - we always want it in output, so training
        on it is unnecessary. Only train on the actual answer content.

        Args:
            input_ids: Token IDs
            mask: Current boolean mask

        Returns:
            Updated mask with answer marker excluded

        Note:
            This is optional. The marker can be useful for parsing output,
            so you may want to keep it in the loss (default behavior).
        """
        marker_tokens = self.answer_marker_tokens  # Precomputed in __init__
        marker_len = len(marker_tokens)

        if marker_len == 0:
            return mask  # No marker to exclude

        # Convert to tensor on same device
        marker_tensor = torch.tensor(
            marker_tokens, device=input_ids.device, dtype=input_ids.dtype
        )

        # Find all occurrences of the marker
        for i in range(len(input_ids) - marker_len + 1):
            if torch.all(input_ids[i : i + marker_len] == marker_tensor):
                # Exclude these positions from mask
                mask[i : i + marker_len] = False

        return mask

    def _validate_mask_sanity(
        self,
        mask: torch.Tensor,
        seq_len: int,
    ) -> None:
        """
        Final sanity check on the mask.

        Catches edge cases like:
        - All tokens masked (no loss computed)
        - Suspiciously few/many tokens masked

        Args:
            mask: Final boolean mask
            seq_len: Sequence length

        Raises:
            AssertionError: If mask looks suspicious
        """
        num_unmasked = mask.sum().item()
        ratio = num_unmasked / seq_len if seq_len > 0 else 0

        # Check 1: At least SOME tokens should be unmasked
        assert (
            num_unmasked > 0
        ), "All tokens are masked! No loss will be computed. Check your data."

        # Check 2: Not ALL tokens should be unmasked (suspicious)
        assert num_unmasked < seq_len, (
            f"No tokens are masked ({num_unmasked}/{seq_len})! "
            f"Check that assistant_only_loss=True in SFTConfig."
        )

        # Check 3: Reasonable ratio (assistant tokens should be 20-80% of sequence)
        if ratio < 0.1:
            logger.warning(
                f"Very few tokens unmasked: {num_unmasked}/{seq_len} ({ratio*100:.1f}%). "
                f"This is unusual. Check your data formatting."
            )
        elif ratio > 0.9:
            logger.warning(
                f"Almost all tokens unmasked: {num_unmasked}/{seq_len} ({ratio*100:.1f}%). "
                f"Are you sure assistant_only_loss=True is working?"
            )

    def _find_answer_start_position(self, input_ids: torch.Tensor) -> int | None:
        """
        Find the starting position of answer marker tokens in sequence.

        Model-agnostic: Searches for tokenized marker, not hardcoded delimiters.

        Args:
            input_ids: [seq_len]

        Returns:
            Position of first token of answer marker, or None if not found
        """
        marker_len = len(self.answer_marker_tokens)

        if marker_len == 0:
            return None

        # Convert marker tokens to tensor on same device
        marker_tensor = torch.tensor(
            self.answer_marker_tokens, device=input_ids.device, dtype=input_ids.dtype
        )

        # Sliding window search
        for i in range(len(input_ids) - marker_len + 1):
            if torch.all(input_ids[i : i + marker_len] == marker_tensor):
                return i

        return None

    def _create_weight_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create per-token loss weight mask using advanced assistant token detection.
        """
        batch_size, seq_len = input_ids.shape
        weight_mask = torch.ones_like(labels, dtype=torch.float32)

        for batch_idx in range(batch_size):
            # Use advanced mask (with sanity checks only on first batch for speed)
            enable_checks = (
                self.enable_mask_sanity_checks
                and batch_idx == 0  # Only check first sample per batch
            )

            assistant_mask = self._get_assistant_token_mask_advanced(
                input_ids[batch_idx],
                labels[batch_idx],
                exclude_answer_marker=self.exclude_answer_marker_from_loss,
                enable_sanity_checks=enable_checks,
            )

            # Find assistant token positions
            assistant_positions = torch.where(assistant_mask)[0]

            if len(assistant_positions) == 0:
                continue  # No assistant tokens in this sample

            assistant_start = assistant_positions[0].item()
            assistant_end = assistant_positions[-1].item() + 1

            # Find answer marker position
            answer_start = self._find_answer_start_position(input_ids[batch_idx])

            if answer_start is not None and answer_start >= assistant_start:
                # Reasoning: from assistant_start to answer_start
                weight_mask[batch_idx, assistant_start:answer_start] = (
                    self.reasoning_weight
                )

                # Answer: from answer_start to end of assistant
                weight_mask[batch_idx, answer_start:assistant_end] = self.answer_weight
            else:
                # No marker found, treat all assistant tokens as reasoning
                weight_mask[batch_idx, assistant_start:assistant_end] = (
                    self.reasoning_weight
                )

            # Ensure masked tokens (label == -100) have 0 weight
            weight_mask[batch_idx][labels[batch_idx] == -100] = 0.0

        return weight_mask

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute weighted loss.

        Model-agnostic: No hardcoded chat template delimiters.
        """
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., :-1].contiguous()

        # Create weight mask (model-agnostic)
        weight_mask = self._create_weight_mask(shift_input_ids, shift_labels)

        # Compute per-token loss
        loss_fct = CrossEntropyLoss(reduction="none")
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
# USAGE EXAMPLES
# ============================================================================

# def example_usage_basic():
#     """Basic usage: Keep marker in loss (recommended for parsing)."""
#
#     trainer = WeightedCoTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["validation"],
#         reasoning_weight=1.0,
#         answer_weight=1.5,
#         answer_marker="Final Answer:",
#         exclude_answer_marker_from_loss=False,  # Keep marker in loss
#         enable_mask_sanity_checks=True,         # Enable validation
#     )
#
#     trainer.train()
#
#
# def example_usage_exclude_marker():
#     """Advanced: Exclude marker from loss."""
#
#     trainer = WeightedCoTTrainerAdvanced(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["validation"],
#         reasoning_weight=1.0,
#         answer_weight=2.0,
#         answer_marker="Final Answer:",
#         exclude_answer_marker_from_loss=True,   # Exclude marker
#         enable_mask_sanity_checks=True,
#     )
#
#     trainer.train()
#
#
# def example_production_mode():
#     """Production: Disable sanity checks for speed."""
#
#     trainer = WeightedCoTTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=dataset["train"],
#         eval_dataset=dataset["validation"],
#         reasoning_weight=1.0,
#         answer_weight=1.5,
#         answer_marker="Final Answer:",
#         exclude_answer_marker_from_loss=False,
#         enable_mask_sanity_checks=False,  # Disable for speed (after validation)
#     )
#
#     trainer.train()
#
#
# if __name__ == "__main__":
#     # Run basic usage
#     example_usage_basic()
