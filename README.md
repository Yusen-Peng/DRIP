<p align="center">
<img src="docs/DRIP_new.png" width="800"/>
</p>

<h1 align="center">DRIP</h1>
<h2 align="center">Dynamic Patch Pooling for Efficient Vision Transformers</h2>

[Full Results on Google Sheet](https://docs.google.com/spreadsheets/d/1jfIsPSpiPZZjCjGudOQiIYASim_LiHTu2kArpMDlIgI/edit?gid=0#gid=0)


## Activate Conda Env

```bash
module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP
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

monitor jobs:

```bash
squeue -u yusenpeng
```

check when the job can start running:

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
