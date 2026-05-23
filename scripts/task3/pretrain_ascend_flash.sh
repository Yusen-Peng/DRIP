#!/bin/bash
#SBATCH --job-name=May23_7B_Fixed_20x_pretrain
#SBATCH --output=May23_7B_Fixed_20x_pretrain.txt
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP_flash
source activate DRIP_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((13000 + RANDOM % 20000))
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0

cd /users/PAS2912/yusenpeng/DRIP/

deepspeed --num_gpus=1 src/task3_llava.py \
    --deepspeed src/LLaVA_wrapper/scripts/mix_free_flash.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /fs/scratch/PAS2836/yusenpeng_dataset/blip_laion_cc_sbu_558k.json \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_pretrain_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_20x_pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --bf16 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
