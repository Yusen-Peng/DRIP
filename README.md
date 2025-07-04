# 💧 DTP-ViT: Dynamic Token Pooling Vision Transformer

## DTP-ViT v.s. existing work

| design | approach summary |
| ------ | -------------------------- |
| **DTP-ViT** (ours!) | a single boundary predictor using **Gumbel-Sigmoid** |
| DynamicViT (2021) | a binary decision mask to **PRUNE** tokens at each transformer layer |
| TokenLearner (2021) | a spatial attention module inserted in ViT to **LEARN** tokens |  
| NativeSegViT (2025) | kmeans-like clustering to dynamically **GROUP** tokens repeatedly |

According to DTP paper, both **Gumbel-Sigmoid** and **Entropy-Spike** are very suitable to adapt to other modalities!

## DTP-ViT Architecture

```txt
input sequence
     ↓
embedding (dropout)
     ↓
pre-layers (# is a HP, 2 default)
     ↓
boundary predictor (MLP)
     ↓
downsampling
     ↓
shortened-layers (# is a HP, 10 default)
     ↓
pooling
     ↓
dense
     ↓
embeddings ready for contrastive learning
```

## Efficiency Metrics

1. GFLOPs: a different script (adapted from **DynamicViT**), NOT during training
     1. a **pretrained** ViT-B-32 is used to compute FLOPs for ViT-B-32
     2. important Adaptations from DynamicViT:
          [FLOP measurement](https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py)
          [simulating artificial boundaries for DynamicViT](https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py)

2. GPU memory and training step time are averaged for each epoch
     1. memory: torch.cuda.max_memory_allocated()
     2. training step time

### Preliminaries: FLOP Analysis

![alt text](docs/FLOP_analysis_plot.png)

### Preliminaries: Boundary Visualization

3 example images from COCO validation set (the model is DTP-VIT-10x pretrained on LAION-27M with 10 epochs):

Sanity check: 

1. we know image: 224x224; patch size: 32
2. thus, boundary mask: (224/32)x(224/32) = 7x7

![alt text](/boundary_visualization_1.png)
![alt text](/boundary_visualization_2.png)
![alt text](/boundary_visualization_3.png)

Note: results won't be deterministic unless we set the random seed (in this case, seed = 42)

## TASK 1 - ImageNet Classification

### Performance Metrics

Classification accuracy on ImageNet

### train ViTs on ImageNet-1K (1.28M images)

reference: zero-shot performance of pretrained CLIPs 

| pretrained vision encoder | corresponding dataset | zero-shot dataset | zero-shot top-1 |
| ------------------------- | --------------------- | ----------------- | --------------- |
| ViT-B-32 | laion2b_s34b_b79k | ImageNet-1K | **66.53%** |

<img src="/lr_finder_plot_DTP_ViT.png" alt="Alt Text" width="500" height="400">

| model | dataset pretrained on | freeze the backbone? | batch size | epoch | classification accuracy |
| ----- | --------------------- | -------------------- | ---------- | ----- | ----------------------- |
| <tr><td colspan="6" align="center"> pretrained ViT </td></tr> |
| ViT-B-32 | laion2b_s34b_b79k | **yes** | 512 | 10 | 🟢75.95% |
| ViT-B-32 | laion2b_s34b_b79k | **yes** | 512 | 30 | 🟢76.81% |
| ViT-B-32 | laion2b_s34b_b79k | finetune all | 512 | 10 | 🟠61.45%: overfitting |
| ViT-B-32 | laion2b_s34b_b79k | finetune all | 512 | 30 | 🟠60.98%: overfitting, train-acc > 96% |
| <tr><td colspan="6" align="center"> train ViT from scratch </td></tr> |
| ViT-B-32 | no initialization | **1e-4 constant scheduler** | 512 | 30 | 🔴**24.02%**: underfitting, train-acc = 24% |
| ViT-B-32 | no initialization | **6e-4 cosine scheduler with warmup** | 512 | 30 | 🔴**35.92%** |
| ViT-B-32 | no initialization | **3.59e-04 cosine scheduler with warmup** | 512 | 100 | 🔴**34.44%** |
| <tr><td colspan="6" align="center"> train DTP-ViT from scratch </td></tr> |
| 10x compression | no initialization | **1e-4 constant scheduler** | 512 | 30 | 🔴**25.43%**: underfitting, train-acc = 24% |
| 10x compression | no initialization | **6e-4 cosine scheduler with warmup** | 512 | 30 | 🔴**24.95%** |
| 10x compression | no initialization | **2.48e-04 cosine scheduler with warmup** | 512 | 100 | 🔴**30.52%** |
| <tr><td colspan="6" align="center"> pretrained DTP-ViT </td></tr> |
| <tr><td colspan="6" align="center"> wait for a **GOOD** pretrained DTP-ViT from CLIP training! </td></tr> |


## TASK 2 - Contrastive Pretraining (CLIP)

### Performance Metrics

Top-1 Acc (%) and Top-5 Acc (%) on ImageNet **Zero-Shot**

### Efficiency Metrics

1. GFLOPs: a different script (adapted from **DynamicViT**), NOT during training
     1. a **pretrained** ViT-B-32 is used to compute FLOPs for ViT-B-32
     2. important Adaptations from DynamicViT:
          [FLOP measurement](https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py)
          [simulating artificial bounddaries for DynamicViT](https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py)

2. GPU memory and training step time are averaged for each epoch.
     1. memory: torch.cuda.max_memory_allocated()
     2. training step time: **already built-in** by CLIP!

### the compression rate is NOT being enforced (or at least controlled)

1. For compression rate = 0.1 at the end of last epoch:

```java
2025-07-03,13:53:18 | INFO | Train Epoch: 9 [26378240/26378240 (100%)] Avg Boundaries (per batch): 9.002 Boundary Ratio: 0.184 Contrastive_loss: 0.80628 (0.83487) Boundary_loss: 0.091194 (0.086909) Loss: 0.89748 (0.92178)
```

2. For compression rate = 0.25 at the end of last epoch:

```java
2025-07-04,09:57:56 | INFO | Train Epoch: 9 [26378240/26378240 (100%)] Avg Boundaries (per batch): 8.723 Boundary Ratio: 0.178 Contrastive_loss: 0.83096 (0.84051) Boundary_loss: 0.071207 (0.073377) Loss: 0.90217 (0.91388)
```

3. For compression rate = 0.5 at the end of last epoch:

```java
2025-07-04,10:51:05 | INFO | Train Epoch: 9 [26378240/26378240 (100%)] Avg Boundaries (per batch): 9.148 Boundary Ratio: 0.187 Contrastive_loss: 0.87118 (0.84179) Boundary_loss: 0.27349 (0.26606) Loss: 1.1447 (1.1079)
```

4. ablation study: For compression rate = 1 (don't compress?!) at the end of last epoch:

```java
2025-07-04,11:31:06 | INFO | Train Epoch: 9 [26378240/26378240 (100%)] Avg Boundaries (per batch): 8.242 Boundary Ratio: 0.168 Contrastive_loss: 0.82972 (0.83114) Boundary_loss: 12.829 (12.800) Loss: 13.659 (13.631)
```

5. ablation study: For compression rate = 0.02 (compress by 50x?!) at the end of last epoch:

```java
2025-07-04,10:56:32 | INFO | Train Epoch: 9 [26378240/26378240 (100%)] Avg Boundaries (per batch): 9.482 Boundary Ratio: 0.194 Contrastive_loss: 0.79891 (0.84947) Boundary_loss: 0.34902 (0.34504) Loss: 1.1479 (1.1945)
```

### LAION-2B subset (26M samples) results

reference: zero-shot performance of pretrained CLIPs 

| pretrained vision encoder | corresponding dataset | zero-shot dataset | zero-shot top-1 |
| ------------------------- | --------------------- | ----------------- | --------------- |
| ViT-B-32 | laion400m_e31 | ImageNet-1K | **60.22%** |

| model | GFLOPs (fvcore) | resolution | patch size | #epochs | Top-1 Acc (%) | Top-5 Acc (%) | avg GPU memory (GB) | avg training step time (s) |
| ------- | ----- | --------------- | ---------- | -------- | ---------- | ---------------- | ------------- | ---------- |
| ViT-B-32 | 2.96 | 224 | 32 | 10 | **28.77%** | 54.34% | 20.1 | 0.429 |
| 2x comp | **2.69** | 224 | 32 | 10 | **21.91%** | 44.86% | **17.6** | **0.393** |
| 4x comp | **1.83** | 224 | 32 | 10 | **21.51%** | 44.25% | **17.7** | **0.391** |
| 10x comp | **1.26** | 224 | 32 | 10 | **21.62%** | 44.64% | **17.6** | **0.395** |

### LAION-400M (?M samples) results

Status up to July 2nd, 12:30AM: **34M** samples processed (over a week)

img2dataset official script - they claim *"400M image/text pairs that can be downloaded in 3.5 days"*: 

```bash
img2dataset \
  --url_list laion400m-meta \
  --input_format "parquet" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_format webdataset \
  --output_folder laion400m-data \
  --processes_count 16 \
  --thread_count 128 \
  --image_size 256 \
  --save_additional_columns '["NSFW","similarity","LICENSE"]' \
  --enable_wandb True
```


my script (using login node - might be the issue?):

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nohup img2dataset \
  --url_list laion400m-meta \
  --input_format "parquet" \
  --url_col "URL" \
  --caption_col "TEXT" \
  --output_format webdataset \
  --output_folder DATASET_LAION400M \
  --processes_count 16 \
  --thread_count 128 \
  --image_size 256 \
  --enable_wandb False \
  --log_level debug \
  > laion400m_download.log 2>&1 &
```

## TASK 3 - Visual Instruction Tuning (LLaVA)

Benchmarks

1. LLaVA-Bench (COCO): COCO-Val-2014, 30 images, 90 questions
2. LLaVA-Bench (In-the-Wild): curate 24 images, 60 questions

### Pretraining - Feature Alignment

dataset (already prepared🔥): **558K** samples 
     
1. image-caption data (LAION-CC-SBU with BLIP captions) -> conversation data
2. objective: image + question -> response
3. both visual encoder and LLM are frozen, only train the projection layer

Pretraining experiment (taking 32 hours with ViT-L-14, vicuna-7B, **1 epoch**):

```T
{'loss': 2.1691, 'grad_norm': 0.6584503898999098, 'learning_rate': 0.0, 'epoch': 1.0}

100%|██████████| 17442/17442 [32:31:42<00:00,  5.80s/it]
                                                        
{'train_runtime': 117102.9646, 'train_samples_per_second': 4.766, 'train_steps_per_second': 0.149, 'train_loss': 2.112418864089378, 'epoch': 1.0}
```

### Visual Instruction Tuning

dataset (LLaVA-Instruct-158K): 158K samples (58K conversations, 23K detailed description, 77K in complex reasoning)

finetuning on (i) multimodal chatbot using LLaVA-Instruct-158K for **3 epochs** (ii) Science QA benchmark; **visual encoder is frozen**, update LLM and the projection layer
