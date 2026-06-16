#!/bin/bash
#SBATCH --job-name=June15_LLaVA_7B_SigLIP_512_DRIP_8x_train_all
#SBATCH --output=June15_LLaVA_7B_SigLIP_512_DRIP_8x_train_all.txt
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=none
#SBATCH --partition=quad
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836
#SBATCH --exclude=a0001,a0002,a0004,a0010,a0011,a0023,a0022

module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP_flash
source activate DRIP_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export WANDB_DISABLED=true
export SLURM_GPU_BIND=none

cd /users/PAS2912/yusenpeng/DRIP/

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

torchrun --standalone --nproc_per_node=4 --master_port=$MASTER_PORT src/task3_llava.py \
    --deepspeed src/LLaVA_wrapper/scripts/finetune_zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning \
    --vision_tower timm/vit_large_patch16_siglip_512.v2_webli \
    --mm_projector_type mlp2x_gelu \
    --tf32 True \
    --bf16 True \
    --pretrain_mm_mlp_adapter /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_pretrain_512_DRIP_8x/mm_projector.bin \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_512_DRIP_8x_train_all \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --accelerator_config '{"gradient_accumulation_kwargs":{"sync_each_batch":true}}' \
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
