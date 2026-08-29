# Qwen-VL finetuning

## Environment

```bash
module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl_flash
```

## SFT


```bash
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune
# uncompressed baseline
sbatch SFT_job.sh
# fixed pooling or DRIP, configure in train/train_compressed_qwen.py
sbatch SFT_job_compressed.sh
```

## Eval


```bash
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune
bash scripts/eval/EVALUATE_ALL.sh
```
