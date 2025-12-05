import logging

from pathlib import Path

from .cwe_filter import Sample
from ..utilities import require_file, validate_jsonl, count_total_lines
from ..datatypes import CWEId

logger = logging.getLogger(__name__)


@require_file("input_file")
@validate_jsonl("input_file")
def filter_unsubstantiated_samples(cot_dataset) -> list[dict]:
    """
    Remove samples where model couldn't substantiate known outcome CWEs
    These are likely labeling errors or edge cases
    """
    clean_samples = []

    for sample in cot_dataset:
        verdict = sample.get('verdict', {})
        unsubstantiated = verdict.get('unsubstantiated_cwes', [])

        # Only keep samples where ALL CWEs were substantiated
        if not unsubstantiated:
            # Remove the unsubstantiated_cwes field before fine-tuning
            if 'unsubstantiated_cwes' in verdict:
                del verdict['unsubstantiated_cwes']
            
            clean_samples.append(sample)
        else:
            # Log what we're filtering
            logger.info(
                f"Filtered sample: Could not substantiate CWEs {unsubstantiated}"
            )
    
    logger.info(
        f"Filtered {len(cot_dataset) - len(clean_samples)} samples "
        f"with unsubstantiated CWEs"
    )
    
    return clean_samples

