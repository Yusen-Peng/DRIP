# command

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



## unzip dataset

```bash
unzip filename.zip -x "__MACOSX/*" "*.DS_Store"
```


