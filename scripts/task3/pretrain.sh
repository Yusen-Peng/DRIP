#!/bin/bash
#SBATCH --job-name=LLaVA_pretrain
#SBATCH --output=LLaVA_pretrain.txt
#SBATCH --time=40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate Fast-CLIP
source activate Fast-CLIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export WANDB_DISABLED=true

cd /users/PAS2912/yusenpeng/Fast-CLIP/

deepspeed src/LLaVA_wrapper/llava_local/train/train_mem.py \
    --deepspeed src/LLaVA_wrapper/scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /fs/scratch/PAS2836/yusenpeng_dataset/blip_laion_cc_sbu_558k.json \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_pretrain_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
