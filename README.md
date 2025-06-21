# 💧 DTP-ViT: Dynamic Token Pooling Vision Transformer

## DTP-ViT v.s. existing work

| design | approach summary |
| ------ | -------------------------- |
| **DTP-ViT** (ours!) | a single boundary predictor using **Gumbel-Sigmoid** |
| DynamicViT (2021) | a binary decision mask to **PRUNE** tokens at each transformer layer |
| TokenLearner (2021) | a spatial attention module inserted in ViT to **LEARN** tokens |  
| NativeSegViT (2025) | kmeans-like clustering to dynamically **GROUP** tokens repeatedly |

According to DTP paper, both **Gumbel-Sigmoid** and **Entropy-Spike** are very suitable to adapt to other modalities:

![alt text](docs/feasible.png)

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

## TASK 1 - ImageNet Classification

### Performance Metrics

Classification accuracy on ImageNet

### Efficiency Metrics

1. GFLOPs: a different script (adapted from **DynamicViT**), NOT during training
     1. a **pretrained** ViT-B-32 is used to compute FLOPs for ViT-B-32
     2. important Adaptations from DynamicViT:
          [FLOP measurement](https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py)
          [simulating artificial bounddaries for DynamicViT](https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py)

2. GPU memory and training step time are averaged for each epoch (**NOT IMPLEMENTED YET**)
     1. memory: torch.cuda.max_memory_allocated()
     2. training step time

### reference: zero-shot performance of pretrained CLIPs 

| pretrained vision encoder | corresponding dataset | zero-shot dataset | zero-shot top-1 |
| ------------------------- | --------------------- | ----------------- | --------------- |
| ViT-B-32 | laion400m_e31 | ImageNet-1K | **60.22%** |
| ViT-B-32 | laion2b_s34b_b79k | ImageNet-1K | **66.53%** |

### train ViTs on ImageNet-1K (1.28M images)

| model | dataset pretrained on | freeze the backbone? | batch size | epoch | zero-shot (as reference) | classification accuracy |
| ----- | --------------------- | -------------------- | ---------- | ----- | ------------------------ | ------------ |
| ViT-B-32 | laion2b_s34b_b79k | **yes** | 512 | 1 | 66.53% | 👍🏻**67.73%** |
| ViT-B-32 | laion2b_s34b_b79k | **yes** | 512 | 10 | 66.53% | TBD |

| 10x compression | **naively** load ALL weights from ViT-B-32 | **yes** | 128 | 1 | 66.53% | 🤡1.46% |
| 10x compression | no initialization (ablation) | **yes** | 128 | 1 | 66.53% | 1.43% |

| ViT-B-32 | laion2b_s34b_b79k | finetune all | 512 | 1 | 66.53% | 50.02% (forgetting) |
| ViT-B-32 | laion2b_s34b_b79k | finetune all | 512 | **10** | 66.53% | running | 


| 10x compression | **naively** load ALL weights from ViT-B-32 | finetune all | 128 | 1 | 66.53% | 16.24% |
| 10x compression | no initialization (ablation) | finetune all | 128 | 1 | 66.53% | 9.66% |


## TASK 2 - Contrastive Pretraining (CLIP-style)

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

### full CC12M (only 7,647,569 - 7M USABLE samples) from img2dataset [official script](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)

| model | GFLOPs (fvcore) | resolution | patch size | #epochs | Top-1 Acc (%) | Top-5 Acc (%) | avg GPU memory (GB) | avg training step time (s) |
| ------- | ----- | --------------- | ---------- | -------- | ---------- | ---------------- | ------------- | ---------- |
| ViT-B-32 | 2.96 | 224 | 32 | **10** | **running** | running | running | running |

| 2x comp | 2.67 | 224 | 32 | **10** |  |  |  |  |
| 4x comp | 1.82 | 224 | 32 | **10** |  |  |  |  |
| 10x comp | 1.25 | 224 | 32 | **10** | **13.38%** | 31.19% | 20.0 | 1.343 |

### LAION-400M (only ? - ?M USABLE samples) from img2dataset [official script](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md)

Pending

## TASK 3 - Visual Instruction Tuning (LLaVA-style)
 
Pending

