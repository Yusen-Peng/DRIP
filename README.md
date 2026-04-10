<p align="center">
<img src="docs/DRIP_new.png" width="800"/>
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

running experiments:

```bash
sbatch scripts/task1/finetune_imagenet.sh
```

boundary visualization & attention map analysis:

for ImageNet

```bash
python src/boundary_visual_IN.py
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
| ![alt text](/src/boundary_vis/w2_4x_boundaries.png)| ![alt text](/src/boundary_vis/w5_4x_attention.png)

## LLaVA

Evaluation:

```bash
# SQA 
sbatch scripts/task3/eval/eval_SQA.sh
# TextVQA
sbatch scripts/task3/eval/eval_textVQA.sh
# POPE
sbatch scripts/task3/eval/eval_POPE.sh
# MM-Bench
sbatch scripts/task3/eval/eval_MMBench.sh
# LLaVA-in-the-wild
sbatch scripts/task3/eval/eval_in_the_wild.sh
```


pretraining/token alignment:

```bash
# ascend
sbatch scripts/task3/pretrain_ascend.sh
# anvil
```

finetuning/VQA SFT:

```bash
sbatch scripts/task3/finetune.sh
```

## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Sachin Kumar (kumar.1145@osu.edu)

Or describe it in Issues.
