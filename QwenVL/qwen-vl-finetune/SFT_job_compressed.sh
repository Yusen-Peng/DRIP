#!/bin/bash
#SBATCH --job-name=Qwen3VL_SFT_Fixed_4x_10data_RERUN
#SBATCH --output=logs/Qwen3VL_SFT_Fixed_4x_10data_RERUN.out
#SBATCH --account=PAS2836
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP_qwenvl_flash
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune
mkdir -p logs


export NPROC_PER_NODE=1
export WORLD_SIZE=1


export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4


# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=128

# Training entry point
entry_file=qwenvl/train/train_compressed_qwen.py

# Dataset configuration
datasets=llava_665k%100

# Output configuration
run_name="qwen3vl_RERUN"
output_dir=/fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_10data_RERUN

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3 \
    --save_total_limit 8 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to "none""

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}