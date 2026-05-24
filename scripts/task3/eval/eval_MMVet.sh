#!/bin/bash
#SBATCH --job-name=May24_MMVet_TEST_lora
#SBATCH --output=May24_MMVet_TEST_lora.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

VERSION="LLaVA_7B_checkpoint_lora"

cd /users/PAS2912/yusenpeng/DRIP/

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-lora-local \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/results
python src/convert_mmvet_for_eval.py \
    --src /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
    --dst /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/results/${VERSION}.json

echo "Evaluation completed for ${VERSION}. Please submit the result json file to: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator"
conda deactivate
# End of script
