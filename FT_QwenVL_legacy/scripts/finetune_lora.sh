#!/bin/bash
#SBATCH --job-name=Feb_20_setup_TEST
#SBATCH --output=Feb_20_setup_TEST.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP-VLM
source activate DRIP-VLM

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))
export WANDB_DISABLED=true


# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))


cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/
# 💥💥💥 absolutely needed! (otherwise, it will cause "ModuleNotFoundError: No module named 'src'") 💥💥💥
export PYTHONPATH=$PWD:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --use_liger_kernel True \
    --lora_enable True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3_offload.json \
    --model_id Qwen/Qwen3-VL-4B-Instruct \
    --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir /fs/scratch/PAS2836/yusenpeng_dataset/testing_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-4 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
