<p align="center">
<img src="docs/DRIP_new.png" width="800"/>
</p>

<h1 align="center">DRIP</h1>
<h2 align="center">Dynamic Patch Pooling for Efficient Vision Transformers</h2>

## Debugging with interactive node

```bash
# simple
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00
# specify the CPU memory when needed
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G
```

## Activate Conda Env

```bash
module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP
```

## GFLOPs measurment

```bash
# DRIP
python src/FLOP.py --mode DRIP --compression_rate 0.25
# Fixed pooling
python src/FLOP.py --mode fixed_pooling --compression_rate 0.25
# original ViT
python src/FLOP.py --mode ViT
```

## ImageNet 

running experiments:

```bash
sbatch scripts/task1/finetune_imagenet.sh
```

boundary visualization & attention map analysis:

for ImageNet

```bash
python src/boundary_visual_IN.py
```


### CLIP pretraining from scratch

```bash
sbatch scripts/task2/multi_gpu_ascend.sh
```

### Result table (classification, CLIP)

| model | IN-configs | IN-Acc | IN-boundaries | CP-configs | CP-Acc | CP-boundaries |
| ----- | ---------- | ------ | ------------- | ---------- | ------ | ------------- |
| vanilla ViT | 100, 0.0003, 0.5 | 47.058% | not bad | 4, 5e-5, 1.0 | 18.07% | bad |
| ViT-RoPE | 100, 0.0003, 0.5 | 65.014% | bad | - | - | - |
| ViT-XL | - | - | - | 4 epochs, 5e-5, temp=0.5 | 20.09% | good |
| ViT-H-net (TBD) | - | - | - | - | - | - |

### LLaVA

pretraining (low-resouce is fine):

```bash
sbatch scripts/task3/pretrain.sh
```

finetuning:

```bash
sbatch scripts/task3/finetune.sh
```


### 🔥🔥NEW🔥🔥 QwenVL series SFT

🧠🧠🧠 finetuning environment 🧠🧠🧠:

```bash
module load miniconda3/24.1.2-py310
conda create -n DRIP-VLM-FT python=3.10 -y
conda activate DRIP-VLM-FT
cd FT_QwenVL/
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
python -m pip install qwen-vl-utils
```

interactive mode for debugging:

```bash
# mem=64G is a must!
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G
# or be specific about partition: for example, quad (sometimes scheduled faster)
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G --partition quad
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM-FT
export PYTHONPATH=$PWD:$PYTHONPATH
# recipe: freeze LLM, train vision encoder (what we need the most)
deepspeed src/train/train_sft.py --use_liger_kernel True --lora_enable True --use_dora False --lora_namespan_exclude "['lm_head', 'embed_tokens']" --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --deepspeed scripts/zero3.json --model_id Qwen/Qwen3-VL-2B-Instruct --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning --remove_unused_columns False --freeze_vision_tower False --freeze_llm True --freeze_merger False --bf16 True --fp16 False --disable_flash_attn2 True --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 32 --image_min_pixels $((224 * 224)) --image_max_pixels $((224 * 224)) --learning_rate 1e-4 --merger_lr 1e-5 --vision_lr 2e-6 --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --gradient_checkpointing True --report_to tensorboard --lazy_preprocess True --save_strategy "steps" --save_steps 200 --save_total_limit 10 --dataloader_num_workers 4
```

skip finetuning by setting num_epochs = 0 -> in order to jump right into eval later:

```bash
# Qwen3VL-4B
deepspeed src/train/train_sft.py --use_liger_kernel True --lora_enable True --use_dora False --lora_namespan_exclude "['lm_head', 'embed_tokens']" --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --deepspeed scripts/zero3.json --model_id Qwen/Qwen3-VL-4B-Instruct --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning --remove_unused_columns False --freeze_vision_tower False --freeze_llm True --freeze_merger False --bf16 True --fp16 False --disable_flash_attn2 True --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora --num_train_epochs 0 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --image_min_pixels $((224 * 224)) --image_max_pixels $((224 * 224)) --learning_rate 1e-4 --merger_lr 1e-5 --vision_lr 2e-6 --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --gradient_checkpointing True --report_to tensorboard --lazy_preprocess True --save_strategy "steps" --save_steps 200 --save_total_limit 10 --dataloader_num_workers 4
# Qwen3VL-2B (smaller)
deepspeed src/train/train_sft.py --use_liger_kernel True --lora_enable True --use_dora False --lora_namespan_exclude "['lm_head', 'embed_tokens']" --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --deepspeed scripts/zero3.json --model_id Qwen/Qwen3-VL-2B-Instruct --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning --remove_unused_columns False --freeze_vision_tower False --freeze_llm True --freeze_merger False --bf16 True --fp16 False --disable_flash_attn2 True --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B --num_train_epochs 0 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --image_min_pixels $((224 * 224)) --image_max_pixels $((224 * 224)) --learning_rate 1e-4 --merger_lr 1e-5 --vision_lr 2e-6 --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --gradient_checkpointing True --report_to tensorboard --lazy_preprocess True --save_strategy "steps" --save_steps 200 --save_total_limit 10 --dataloader_num_workers 4
```

launch official experiment:

```bash
sbatch FT_QwenVL/scripts/finetune_lora.sh
```

fixed pooling - no SFT:

```bash
deepspeed src/train/train_sft.py --use_liger_kernel True --lora_enable True --use_dora False --lora_namespan_exclude "['lm_head', 'embed_tokens']" --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --deepspeed scripts/zero3.json --model_id Qwen/Qwen3-VL-4B-Instruct --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning --remove_unused_columns False --freeze_vision_tower False --freeze_llm True --freeze_merger False --bf16 True --fp16 False --disable_flash_attn2 True --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/fixed_4x_lora --num_train_epochs 0 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --image_min_pixels $((224 * 224)) --image_max_pixels $((224 * 224)) --learning_rate 1e-4 --merger_lr 1e-5 --vision_lr 2e-6 --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 False --gradient_checkpointing True --report_to tensorboard --lazy_preprocess True --save_strategy "steps" --save_steps 200 --save_total_limit 10 --dataloader_num_workers 4 --pooling_strategy Fixed --compression_rate 0.25
```


fixed pooling - with SFT:

TBD


## Benchmarks

[IMPORTANT] before running any benchmark, we need to merge the LoRA checkpoint:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/mmmu
# remember to change HF id, path etc!
python merge_lora.py
```

### Benchmarks - MMMU

we perform inference first (let's use ``Ascend`` for this one):

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:20:00 --mem=64G
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/mmmu
```

The `vLLM` version (from official Qwen3VL codebase):

```bash
python run_mmmu.py infer \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B_merged \
    --data-dir /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval \
    --dataset MMMU_DEV_VAL \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/mmmu_results/predictions.jsonl \
    --max-new-tokens 2048 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5
```

The `HF Transformer` version (adapted from MMMU codebase) -

val split only:

```bash
# NOTE: schedule for at least 70 mins!
python run_mmmu_hf.py \
  --output_path qwen_mmmu_val.json \
  --config_path configs/llava1.5.yaml \
  --data_path MMMU/MMMU \
  --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B_merged \
  --split validation \
  --bf16 \
  --attn_implementation sdpa \
  --max_new_tokens 128 \
  --temperature 0 \
  --pooling_strategy Original \
  --compression_rate 1.0
```

val + dev split:

```bash
# NOTE: schedule for at least 80 mins for this version!
python run_mmmu_hf.py \
  --output_path qwen_mmmu_mixed.json \
  --config_path configs/llava1.5.yaml \
  --data_path MMMU/MMMU \
  --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B_merged \
  --split mixed \
  --bf16 \
  --attn_implementation sdpa \
  --max_new_tokens 128 \
  --temperature 0 \
  --pooling_strategy Original \
  --compression_rate 1.0
```



Then we can finally do evaluation (vLLM):

```bash
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/mmmu
export CHATGPT_DASHSCOPE_API_KEY=<api_key>
# use This API endpoint!! We are in America 🇺🇸🇺🇸
export DASHSCOPE_API_BASE="https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
python run_mmmu.py eval \
    --data-dir /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval \
    --input-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/mmmu_results/predictions.jsonl \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/mmmu_results/evaluation.csv \
    --dataset MMMU_DEV_VAL \
    --eval-model qwen3-max \
    --api-type dash \
    --nproc 16
```

eval with HF transformers:

```bash
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/mmmu
# val only
python eval_mmmu_hf.py --pred_path qwen_mmmu_val.json
# or dev + val
python eval_mmmu_hf.py --pred_path qwen_mmmu_mixed.json --split mixed
```


### Benchmarks - RealWorldQA

inference:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/RealWorldQA
python run_realworldqa.py infer \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B_merged \
    --data-dir /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval \
    --dataset RealWorldQA \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/realworldqa_results/predictions.jsonl \
    --max-new-tokens 2048 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5
```

eval:

```bash
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/RealWorldQA
export CHATGPT_DASHSCOPE_API_KEY=<api_key>
# use This API endpoint!! We are in America 🇺🇸🇺🇸
export DASHSCOPE_API_BASE="https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
python run_realworldqa.py eval \
    --data-dir /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval \
    --input-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/realworldqa_results/predictions.jsonl \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/realworldqa_results/evaluation.csv \
    --dataset RealWorldQA \
    --eval-model qwen3-max \
    --api-type dash \
    --nproc 16
```

### Benchmarks - MathVision

infer:

```bash
# math problems are hard (therefore time-consuming)
# make sure to schedule for 6 hours on pitzer!
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 6:00:00 --mem=64G
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/MathVision
# 8192
python run_mathv.py infer \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_merged \
    --data-dir /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval \
    --dataset MathVision \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/mathvision_results/predictions.jsonl \
    --max-new-tokens 2048 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --repetition-penalty 1.0 \
    --presence-penalty 1.5 \
    --max-model-len 8192
```

eval:


```bash
module load miniconda3/24.1.2-py310
conda activate DRIP-VLM
export PYTHONPATH=$PWD:$PYTHONPATH
export CHATGPT_DASHSCOPE_API_KEY=<api_key>
# use This API endpoint!! We are in America 🇺🇸🇺🇸
export DASHSCOPE_API_BASE="https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions"
cd /users/PAS2912/yusenpeng/Fast-CLIP/FT_QwenVL/benchmarks/MathVision
python run_mathv.py eval \
    --data-dir /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval \
    --input-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/mathvision_results/predictions.jsonl  \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/Qwen_eval/mathvision_results/evaluation.csv \
    --dataset MathVision \
    --eval-model qwen3-max \
    --api-type dash \
    --nproc 16
```


## result table

| model | finetuning | config details | MMMU | RealWorldQA | MathVision | MMBench | ScienceQA | POPE | TextVQA | MME |
| ----- | ---------- | ------- | ---- | ----------- | ---------- | ------- | --------- | ---- | ------- | --- |
| ***original VLMs*** |
| Qwen3VL-4B [paper] | - | - | 67.4% | 70.9% | 51.6% |
| Qwen3VL-4B | no, just eval | - | 66.57% (vLLM) | 70.98% (vLLM) | 47.07% (vLLM) |
| Qwen3VL-2B [paper] | - | - | **53.4%** | 63.9% | 31.6% |
| Qwen3VL-2B | no, just eval | - | dev + validation (1050 samples); Qwen3 judge; 54.38% (vLLM) | 65.88% (vLLM) | TBD |
| Qwen3VL-2B | no, just eval | - | validation split (900 samples); rule-based only; 41.22% (HF) | ?? | TBD |
| Qwen3VL-2B | no, just eval | - | dev + validation (1050 samples); rule-based only; 41.90% (HF) | ?? | TBD |
| Qwen3VL-2B | 1 epoch on LLaVA | lora LLM, full FT ViT | submitted | - | - |
| ***fixed pooling baselines*** |
| Qwen3VL-4B-fixed-4x | no, just eval | - | ? | ? | ? |
| Qwen3VL-4B-fixed-10x | no, just eval | - | ? | ? | ? |
| Qwen3VL-2B-fixed-4x | no, just eval | - | ? | ? | ? |
| Qwen3VL-2B-fixed-10x | no, just eval | - | ? | ? | ? |
| Qwen3VL-2B-fixed-4x | 1 epoch on LLaVA | lora LLM, full FT ViT | ? | ? | ? |
| Qwen3VL-2B-fixed-10x | 1 epoch on LLaVA | lora LLM, full FT ViT | ? | ? | ? |
| ***DRIP*** |
| Qwen3VL-2B-DRIP-4x | 1 epoch on LLaVA | lora LLM, full FT ViT | ? | ? | ? |
| Qwen3VL-2B-DRIP-10x | 1 epoch on LLaVA | lora LLM, full FT ViT | ? | ? | ? |

## Handy Commands

monitor all jobs:

```bash
squeue -u yusenpeng
```

check when a specific job can start running:

```bash
squeue --start -j <JOB_ID>
```

## GPU usage check

```bash
squeue -p quad -o "%.18i %.9P %.20j %.15u %.2t %.10M %.6D %R"
squeue -p nextgen -o "%.18i %.9P %.20j %.15u %.2t %.10M %.6D %R"
```


## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Sachin Kumar (kumar.1145@osu.edu)

Or describe it in Issues.
