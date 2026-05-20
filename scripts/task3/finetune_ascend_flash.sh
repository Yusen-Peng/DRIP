#!/bin/bash
#SBATCH --job-name=May20_7B_Fixed_10x_finetune
#SBATCH --output=May20_7B_Fixed_10x_finetune.txt
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP_flash
source activate DRIP_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export WANDB_DISABLED=true

cd /users/PAS2912/yusenpeng/DRIP/

deepspeed src/task3_llava.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed src/LLaVA_wrapper/scripts/finetune.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tf32 True \
    --bf16 True \
    --pretrain_mm_mlp_adapter /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_10x_pretrain/mm_projector.bin \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_10x_finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 15 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
