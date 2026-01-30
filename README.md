<p align="center">
<img src="docs/DRIP_new.png" width="800"/>
</p>

<h1 align="center">DRIP</h1>
<h2 align="center">Dynamic Patch Pooling for Efficient Vision Transformers</h2>

<!-- ## News/Updates

- [October 27, 2025] model checkpoints pretrained on CLIP/BioCLIP are publicly available on [HuggingFace](https://huggingface.co/YusenPeng/DRIP_checkpoints) for further finetuning! -->

<!-- ## Environment Setup

```bash
conda create -n DRIP python=3.11
conda activate DRIP
python -m pip install open_clip_torch
python -m pip install 'open_clip_torch[training]'
conda install -c conda-forge sentencepiece
python -m pip install braceexpand
python -m pip install webdataset
python -m pip install tensorboard
python -m pip install pdbpp
``` -->

[Full Results on Google Sheet](https://docs.google.com/spreadsheets/d/1jfIsPSpiPZZjCjGudOQiIYASim_LiHTu2kArpMDlIgI/edit?gid=0#gid=0)


## Activate Conda Env

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00 # optionally schedule a job
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


<!-- ## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Sachin Kumar (kumar.1145@osu.edu)

Or describe it in Issues. -->

<!-- ## find the model

```
(DRIP) [yusenpeng@ascend-login01 Fast-CLIP]$ python siglip_explore.py
torch.Size([1, 1152])
Model class: <class 'transformers.models.siglip.modeling_siglip.SiglipModel'>
Module: transformers.models.siglip.modeling_siglip
Source file: /users/PAS2912/yusenpeng/.conda/envs/DRIP/lib/python3.11/site-packages/transformers/models/siglip/modeling_siglip.py
``` -->
