<p align="center">
<img src="docs/DRIP_new.png" width="800"/>
</p>

<h1 align="center">DRIP</h1>
<h2 align="center">Dynamic Patch Pooling for Efficient Vision Transformers</h2>

### Debugging with interactive node

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00
```

## Activate Conda Env

```bash
module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP
# a simple imagenet example (smaller batch size, single worker)
torchrun --nproc_per_node=1 src/task1_newcodebase.py --model vit_b_16 --epochs 300 --batch-size 32 --opt adamw --lr 0.0003 --wd 0.3 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30  --workers 1 --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --output-dir /fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_ViT_RP --MODE ViT-RP
```


## Experiments

### ImageNet from scratch

```bash
sbatch scripts/task1/finetune_imagenet.sh
```

### CLIP pretraining from scratch

```bash
sbatch scripts/task2/multi_gpu_ascend.sh
```

### LLaVA

pretraining (low-resouce is fine):

```bash
sbatch scripts/task3/pretrain.sh
```

finetuning:

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
