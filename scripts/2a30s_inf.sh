#!/bin/bash
#SBATCH --job-name=CoT-INF
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a30.24:2
#SBATCH --output=stdout/%x-%j.out
#SBATCH --error=stderr/%x-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --partition edu-thesis

module load CUDA/12.5.0
module load Python/3.12.3-GCCcore-13.3.0

source vullm/bin/activate

export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
mkdir -p $TRITON_CACHE_DIR

echo "✅ Triton cache: $TRITON_CACHE_DIR"

python DoneBot/src/notify.py --cmd accelerate launch --config_file accelerate/2gpus.yaml -m src.core.cot_training.main --inference --formatted_dataset_dir ./DiverseVul/formatted/reasonings --lora_weights ./checkpoints/best_model --max_seq_length 4096 --max_tokens_per_answer 2048 --assets_dir ./results/experiment_2025-10-27_08-51 --include_code_in_report --save_summary
