#!/bin/bash
#SBATCH --job-name=CoT-Qwen
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
mkdir -p $TRITON_CACHE_DIR

echo "✅ Triton cache: $TRITON_CACHE_DIR"

#"unsloth/llama-3.1-8b-instruct-bnb-4bit",
#"unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
#"unsloth/Qwen3-32B-unsloth-bnb-4bit",
#"unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
#"unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
#"unsloth/QwQ-32B-unsloth-bnb-4bit"

python DoneBot/src/notify.py --cmd accelerate launch --config_file accelerate/2gpus.yaml -m src.core.cot_training.main --finetune --dataset_path ./DiverseVul/processed/gpt_reasoning.jsonl --formatted_dataset_dir ./DiverseVul/formatted/reasonings --base_model_name unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit --chat_template qwen-2.5 --max_seq_length 5120 --epochs 3 --batch_size 2 --grad_acc_steps 4 --learning_rate 5e-5 --use_rslora --deepspeed
