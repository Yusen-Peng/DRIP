# Commands to run experiment

Evaluating a pretrained CLIP:

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --time 0:15:00
module load miniconda3/24.1.2-py310
conda deactivate
conda activate Fast-CLIP
python src/task3_llava.py
python src/eval_CLIP.py
module load cuda/12.4.1
python -m pip install flash-attn --no-build-isolation
```

Training a CLIP from scratch:

```bash
module load miniconda3
conda deactivate
sbatch ./real_run.sh

```

Training a DTP from scratch:

```bash
module load miniconda3
conda deactivate
sbatch ./dtp_run.sh

```

Monitor my jobs:

```bash
squeue -u yusenpeng
```

More details:

```bash
scontrol show job [JOB_ID]
```

to early cancel a job (something is already wrong)

```bash
scancel [JOB_ID]
```

look up all partitions on a cluster

```bash
sinfo -o "%P"
```

## HuggingFace Login

```bash
huggingface-cli login --token [your token]
```

## check disk usage

```bash
quota -s
```

## Who the heck is using the GPUs?

```bash
squeue -p quad -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

## unzip datasets

```bash
unzip filename.zip -x "__MACOSX/*" "*.DS_Store"
```

## check img2dataset process

```bash
ps aux | grep img2dataset
```

## Environment Setup

```bash
conda create -n Fast-CLIP python=3.11
conda activate Fast-CLIP
python -m pip install open_clip_torch
python -m pip install 'open_clip_torch[training]'
conda install -c conda-forge sentencepiece
python -m pip install braceexpand
python -m pip install webdataset
python -m pip install tensorboard
python -m pip install pdbpp
```
