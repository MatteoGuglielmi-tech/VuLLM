"""
CLI tool for analyzing sequence lengths in CoT datasets.

Usage:
    python analyze_sequence_lengths.py \\
        --dataset path/to/dataset.jsonl \\
        --tokenizer unsloth/llama-3-8b-bnb-4bit \\
        --output ./analysis_results \\
        --max-samples 1000

    # Compare multiple tokenizers
    python analyze_sequence_lengths.py \\
        --dataset path/to/dataset.jsonl \\
        --tokenizers unsloth/llama-3-8b-bnb-4bit mistralai/Mistral-7B-Instruct-v0.2 \\
        --output ./comparison
"""
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from sequence_length_analyzer import SequenceLengthAnalyzer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token distribution in CoT dataset to determine optimal max_seq_length"
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset file"
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Single tokenizer to use (e.g., unsloth/llama-3-8b-bnb-4bit)"
    )
    
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        help="Multiple tokenizers to compare"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./sequence_analysis"),
        help="Output directory for results (default: ./sequence_analysis)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)"
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (uses default if not provided)"
    )
    
    parser.add_argument(
        "--prompt-skeleton",
        type=str,
        default=None,
        help="Custom prompt skeleton (uses default
