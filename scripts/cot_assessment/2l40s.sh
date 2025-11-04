#!/bin/bash
#SBATCH --job-name=court
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:2
#SBATCH --output=stdout/%x-%j.out
#SBATCH --error=stderr/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition edu-thesis

module load CUDA/12.5.0
module load Python/3.12.3-GCCcore-13.3.0

source vullm/bin/activate

export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

echo "✅ Triton cache: $TRITON_CACHE_DIR"

# Prevent timeout
export NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1

python DoneBot/src/notify.py --cmd accelerate launch --config_file accelerate/2gpus.yaml -m src.core.cot.assessment.main --input ./DiverseVul/processed/gpt_reasoning.jsonl --output ./DiverseVul/assessed/ --assets ./src/core/cot/assessment/assets/ --max_new_tokens 512 --quality_threshold 0.8 --agreement_threshold 0.75 --agreement_method weighted_multidimensional
