<p align="center">
<img src="docs/DRIP.png" width="700"/>
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

activate an existing one:

```bash
module load miniconda3/24.1.2-py310 # for OSC
module load conda # for Anvil
conda deactivate
conda activate DRIP
```

## ImageNet (OSC pitzer)

running training experiments:

```bash
sbatch scripts/task1/finetune_imagenet.sh
```

boundary visualization & attention map analysis:

for ImageNet

```bash
python src/boundary_visual_IN.py
```

running eval + analysis experiments:

```bash
sbatch scripts/task1/eval_imagenet.sh
```

GFLOPs measurement:

```bash
# DRIP
python src/FLOP.py --mode DRIP --compression_rate 0.25
# Fixed pooling
python src/FLOP.py --mode fixed_pooling --compression_rate 0.25
# original ViT
python src/FLOP.py --mode ViT
```

examples:

| boundaries | attention maps |
| ---------- | -------------- |
| imagenet_DRIP_4x_01_warmup2 | model_299.pth |
| ![alt text](/src/boundary_vis/w2_4x_boundaries.png) | ![alt text](/src/boundary_vis/w2_4x_attention.png) |
<!-- | imagenet_DRIP_4x_half_LR_no_warmup | model_299.pth |
| ![alt text](/src/boundary_vis/halfLR_dtpvit_single_multi_overlay.png) | ![alt text](/src/boundary_vis/halfLR_dtpvit_single_multi_attention_overlay.png) | -->


## LLaVA

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

General VQA (4):

```bash
# SQA 
sbatch scripts/task3/eval/eval_SQA.sh
# MM-Bench
sbatch scripts/task3/eval/eval_MMBench.sh
# MME
sbatch scripts/task3/eval/eval_MME.sh
# VQAv2 [🚨LONG🚨]
# need to submit the result json file to:
# https://eval.ai/web/challenges/challenge-page/830
sbatch scripts/task3/eval/eval_VQAv2.sh
```

Reasoning (1):

```bash
# GQA
sbatch scripts/task3/eval/eval_GQA.sh
```

OCR (1):

```bash
# TextVQA
sbatch scripts/task3/eval/eval_textVQA.sh
```

Hallucination (1):

```bash
# POPE
sbatch scripts/task3/eval/eval_POPE.sh
```

Free Response (1):

```bash
# LLaVA-in-the-wild
sbatch scripts/task3/eval/eval_in_the_wild.sh
```

## LLaVA Finetuning

### flash attention

Before anything, make sure flash attention is installed:

```bash
# install
sbatch flash_attn.sh
# test
sbatch test_flash_attn.sh
# what to expect: 
# torch.Size([1, 128, 8, 64]) torch.float16 cuda:0
```

### pretraining (token alignment)

```bash
# ascend with flash attention
sbatch scripts/task3/pretrain_ascend_flash.sh
```

When resuming from an existing checkpoint, **make sure** to update the DRIP weight path `DRIP_WEIGHT_PATH` accordingly:

```python
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
```

### finetuning/VQA SFT

```bash
# ascend with flash attention
sbatch scripts/task3/finetune_ascend_flash.sh
```

When resuming from an existing checkpoint, **make sure** to update the DRIP weight path `DRIP_WEIGHT_PATH`

```python
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train/checkpoint-1020/drip.bin"
```

**AND** the MLP projector path in the SLURM scripts:

```bash
--pretrain_mm_mlp_adapter /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train/checkpoint-1020/mm_projector.bin \
```


## LLaVA boundary visualization

For LLaVA visualization, a GPU is definietely needed:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:15:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/boundary_visual_LLaVA.py
```

### visualization examples

2x compression after llava pretraining:

![alt text](src/boundary_vis/LLaVA_results/2x_pretrain_llava_drip_boundaries_2x5.png)

4x compression after llava pretraining:

![alt text](src/boundary_vis/LLaVA_results/4x_pretrain_llava_drip_boundaries_2x5.png)

4x compression after llava finetuning (2661 steps for now):

![alt text](src/boundary_vis/LLaVA_results/4x_2661_sft_llava_drip_boundaries_2x5.png)

4x compression after full llava finetuning:

TBD


### checking if BP changes

```bash
python - <<'PY'
import torch
a_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
b_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train/checkpoint-1020/drip.bin"

def normalize(sd):
    out = {}
    for k, v in sd.items():
        if "vision_tower.boundary_predictor." in k:
            nk = k.split("vision_tower.boundary_predictor.", 1)[1]
            out[nk] = v.detach().float().cpu()
        elif k.endswith("vision_tower.null_token") or k == "null_token":
            out["null_token"] = v.detach().float().cpu()
    return out
a = normalize(torch.load(a_path, map_location="cpu"))
b = normalize(torch.load(b_path, map_location="cpu"))
print("keys A:", sorted(a.keys()))
print("keys B:", sorted(b.keys()))
for k in sorted(set(a) & set(b)):
    x, y = a[k], b[k]
    d = y - x
    print(f"\n{k}")
    print(f"  A norm:     {x.norm().item():.6g}")
    print(f"  B norm:     {y.norm().item():.6g}")
    print(f"  diff norm:  {d.norm().item():.6g}")
    print(f"  rel diff:   {(d.norm() / (x.norm() + 1e-12)).item():.6g}")
    print(f"  max abs:    {d.abs().max().item():.6g}")
    print(f"  allclose:   {torch.allclose(x, y)}")
PY
```


## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Sachin Kumar (kumar.1145@osu.edu)

Or describe it in Issues.
