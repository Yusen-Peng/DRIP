#!/bin/bash
#SBATCH --job-name=0826_OCRBenchv2_Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e3_2xwidth
#SBATCH --output=0826_OCRBenchv2_Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e3_2xwidth.log
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl_flash


export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

VERSION="Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e3_2xwidth"

# python eval_code/model_vqa_ocrbenchv2.py \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --dataset_path lmms-lab/OCRBench-v2 \
#     --dataset_split test \
#     --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
#     --save_name ${VERSION} \
#     --num_workers 1 \
#     --temperature 0


# python eval_code/model_vqa_ocrbenchv2.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL/checkpoint-18 \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --dataset_path lmms-lab/OCRBench-v2 \
#     --dataset_split test \
#     --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
#     --save_name ${VERSION} \
#     --num_workers 1 \
#     --temperature 0


# python eval_code/model_vqa_ocrbenchv2.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_2x_10data \
#     --model_base Qwen/Qwen3-VL-4B-Instruct \
#     --dataset_path lmms-lab/OCRBench-v2 \
#     --dataset_split test \
#     --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
#     --save_name ${VERSION} \
#     --num_workers 1 \
#     --temperature 0 \
#     --merge-strategy Fixed \
#     --compression-rate 0.5


python eval_code/model_vqa_ocrbenchv2.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e3_2xwidth \
    --model_base Qwen/Qwen3-VL-4B-Instruct \
    --dataset_path lmms-lab/OCRBench-v2 \
    --dataset_split test \
    --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
    --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
    --save_name ${VERSION} \
    --num_workers 1 \
    --temperature 0 \
    --merge-strategy DRIP \
    --compression-rate 0.25 \
    --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e3_2xwidth/drip.bin



conda deactivate

#### Evaluation ####
####################

cd /users/PAS2912/yusenpeng/DRIP

conda activate OCRv2Eval


#### step1: merge json files

python src/OCRBenchv2_eval/json_merge.py \
    --pred /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results/${VERSION}.json \
    --out /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results/${VERSION}_merged.json

#### step2: run evaluation

export NLTK_DATA=/fs/scratch/PAS2836/yusenpeng_dataset/nltk_data
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores

python src/OCRBenchv2_eval/eval.py \
    --input_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results/${VERSION}_merged.json \
    --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores/${VERSION}_scores.json

python src/OCRBenchv2_eval/get_score.py \
    --json_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores/${VERSION}_scores.json

conda deactivate
