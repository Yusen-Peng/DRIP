#!/bin/bash
#SBATCH --job-name=Nov4_pretrain_drip_10epochs
#SBATCH --output=Nov4_pretrain_drip_10epochs.txt
#SBATCH --time=64:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export WANDB_DISABLED=true

cd /users/PAS2912/yusenpeng/Fast-CLIP/

deepspeed src/task3_llava.py \
    --deepspeed src/LLaVA_wrapper/scripts/mix_free.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /fs/scratch/PAS2836/yusenpeng_dataset/blip_laion_cc_sbu_558k.json \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_pretrain_images \
    --vision_tower openai/clip-vit-base-patch16 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/ViTbased-DRIP-4x-16-4-8-pretrain-10-epochs \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
