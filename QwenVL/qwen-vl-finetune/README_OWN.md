# Qwen-VL finetuning

## Environment

```bash
module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl_flash
```

## SFT


```bash
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune
sbatch SFT_job.sh
```


## Eval


```bash
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune
sbatch sbatch scripts/eval/TextVQA.sh
sbatch scripts/eval/DocVQA.sh
sbatch scripts/eval/OCRBench.sh
sbatch scripts/eval/OCRBenchv2.sh
sbatch scripts/eval/ChartQAPro.sh
```
