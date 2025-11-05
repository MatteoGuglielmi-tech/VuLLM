#!/bin/bash
#SBATCH --job-name=court
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
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

python DoneBot/src/notify.py \
	--cmd python -m src.core.cot.assessment.main \
	--sequential \
	--input ./DiverseVul/processed/safe.jsonl \
	--output_path ./DiverseVul/assessed/llama_safe.jsonl \
	--judge llama-3.1-70B \
	--max_lengths 4096 4096 4096 \
	--max_new_tokens 256 \
	--save_interval 500
