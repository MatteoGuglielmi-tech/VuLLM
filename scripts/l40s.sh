#!/bin/bash
#SBATCH --job-name=CoT-PEFT
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

nvidia-smi

python DoneBot/src/notify.py --cmd accelerate launch --config_file accelerate/2gpus.yaml -m src.core.cot_training.main --finetune --dataset_path ./DiverseVul/processed/gpt_reasoning.jsonl --formatted_dataset_dir ./DiverseVul/formatted/reasonings --max_seq_length 4096 --epochs 3 --batch_size 4 --grad_acc_steps 8 --use_rslora --deepspeed
