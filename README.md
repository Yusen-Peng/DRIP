<p align="center">
<img src="docs/DRIP_new.png" width="800"/>
</p>

<h1 align="center">DRIP</h1>
<h2 align="center">Dynamic Patch Pooling for Efficient Visual Instruction Tuning</h2>

## Activate Conda Env

```bash
# optional GPU
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00
module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP
```

## GFLOPs measurement

```bash
# DRIP
python src/FLOP.py --mode DRIP --compression_rate 0.25
# Fixed pooling
python src/FLOP.py --mode fixed_pooling --compression_rate 0.25
# original ViT
python src/FLOP.py --mode ViT
```

## ImageNet 

running experiments (make sure to use **OSC pitzer** cluster):

```bash
sbatch scripts/task1/finetune_imagenet.sh
```

boundary visualization & attention map analysis (**no GPU** is needed):

for ImageNet

```bash
python src/boundary_visual_IN.py
```

## LLaVA

pretraining/token alignment (MUST use **ascend** cluster):

```bash
sbatch scripts/task3/pretrain.sh
```

finetuning/VQA SFT (MUST use **ascend** cluster):

```bash
sbatch scripts/task3/finetune.sh
```

## Handy Commands

monitor all jobs:

```bash
squeue -u yusenpeng
```

check when a specific job can start running:

```bash
squeue --start -j <JOB_ID>
```

## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Sachin Kumar (kumar.1145@osu.edu)

Or describe it in Issues.
