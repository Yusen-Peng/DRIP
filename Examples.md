# Example Analysis


## TextVQA

Find examples in which DRIP gets correct answer but fixed pooling fails:

```bash
python src/example_analysis/TextVQA_analysis.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/TextVQA_0.5.1_val.json \
    --fixed-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/LLaVA_7B_Fixed_4x_finetune_train_lora.jsonl \
    --drip-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/LLaVA_7B_DRIP_4x_finetune_train_lora.jsonl
```

Visualize these examples by configuring the image ID one by one:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:05:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/example_analysis/individual_boundaries.py
```


## OCRBench


## OCRBench v2


## DocVQA


## ChartQAPro

