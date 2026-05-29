<p align="center">
<img src="assets/DRIP-pipeline.png" width="500" height="400"/>
</p>

<h1 align="center">DRIP</h1>
<h2 align="center">Dynamic Patch Pooling for Efficient Visual Instruction Tuning</h2>

## Environment setup

Create a new conda enviornment from scratch:

```bash
module load miniconda3/24.1.2-py310 # for OSC
module load conda # for Anvil
conda create -n DRIP python=3.11 -y
conda activate DRIP
python -m pip install -r requirements.txt
```

## LLaVA 1.5 Experiments

### Instruction

Go to file [`src/LLaVA_wrapper/llava_local/model/multimodal_encoder/builder.py`](src/LLaVA_wrapper/llava_local/model/multimodal_encoder/builder.py) to configure merging strategies and corresponding compression rate:

```python
MERGE_STRATEGY = "DRIP" # "ViT" or "DRIP" or "Fixed" or "PruMerge"
COMPRESSION_RATE = 0.25
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
```

Additional note: the ViT backbone from LLaVA checkpoint is `openai/clip-vit-large-patch14-336`.

Then we are good to move onto benchmark experiments.

### Evaluation/Benchmarks

Do evaluation across all **14** VQA benchmarks:

```bash
bash scripts/task3/eval/EVALUATE_ALL.sh
```

## LLaVA Finetuning

Before anything, make sure flash attention is installed - instruction: [https://github.com/Yusen-Peng/EasyInstall#flash-attention](https://github.com/Yusen-Peng/EasyInstall#flash-attention).

### pretraining (token alignment)

```bash
# ascend with flash attention
sbatch scripts/task3/pretrain_ascend_flash.sh
```

When resuming from an existing checkpoint, **make sure to update the DRIP weight path `DRIP_WEIGHT_PATH` accordingly**:

```python
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
```

### finetuning/VQA SFT

We use ascend cluster with flash attention:

```bash
# LoRA finetuning - single GPU is fine
sbatch scripts/task3/finetune_ascend_flash.sh
# Full finetuning - must be distributed
# 2 GPUs OR 4 GPUs
sbatch scripts/task3/finetune_ascend_flash_full.sh
```

We can SSH into GPUs to check its memory usage with:

```bash
ssh <node ID> nvidia-smi
```

and process status with:

```bash
ssh <node ID> "ps -fp <job ID>"
```

When resuming from an existing checkpoint, **make sure to update the DRIP weight path `DRIP_WEIGHT_PATH`**

```python
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train/checkpoint-1020/drip.bin"
```

**AND the MLP projector path in the SLURM scripts**:

```bash
--pretrain_mm_mlp_adapter /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train/checkpoint-1020/mm_projector.bin \
```


## LLaVA boundary visualization

For LLaVA visualization, a GPU is definietely needed:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:05:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/boundary_visual_LLaVA.py
```

You can find examples in [Boundaries.md](/Boundaries.md). You can also find interesting image feature analysis (PCA, CLS attention, cosine similarity) in [Features.md](/Features.md).


## Benchmark Results

[Overleaf Link (restricted)](https://www.overleaf.com/project/69d110e27f4d521bbd6449ec)

## Contact

Yusen Peng (peng.1007@osu.edu)

