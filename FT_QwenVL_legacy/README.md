# Qwen3VL finetuning handbook

## Environment

```bash
module load miniconda3/24.1.2-py310
conda create -n DRIP_qwenvl python=3.10 -y
conda activate DRIP_qwenvl
cd FT_QwenVL/
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu128
python -m pip install qwen-vl-utils
```
interactive mode for debugging:

```bash
# mem=64G is a must!
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G
# or be specific about partition: for example, quad (sometimes scheduled faster)
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 --mem=64G --partition debug-quad



salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:25:00
cd /users/PAS2912/yusenpeng/DRIP/FT_QwenVL/
module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl
export PYTHONPATH=$PWD:$PYTHONPATH
```

skip finetuning by setting num_epochs = 0 -> in order to jump right into eval later:

```bash
# Qwen3VL-4B
deepspeed src/train/train_sft.py --use_liger_kernel True --lora_enable True --use_dora False --lora_namespan_exclude "['lm_head', 'embed_tokens']" --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --num_lora_modules -1 --deepspeed scripts/zero3.json --model_id Qwen/Qwen3-VL-4B-Instruct --data_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning --remove_unused_columns False --freeze_vision_tower True --freeze_llm True --freeze_merger True --bf16 True --fp16 False --disable_flash_attn2 True --output_dir /fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora --num_train_epochs 0 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --image_min_pixels $((224 * 224)) --image_max_pixels $((224 * 224)) --learning_rate 1e-4 --merger_lr 1e-5 --vision_lr 2e-6 --weight_decay 0.1 --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 True --gradient_checkpointing True --report_to tensorboard --lazy_preprocess True --save_strategy "steps" --save_steps 200 --save_total_limit 10 --dataloader_num_workers 4



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
