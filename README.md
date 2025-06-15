# DTP-ViT: Dynamic Token Pooling Vision Transformer for Efficient Contrastive Pretraining

## DTP-ViT v.s. existing work

| design | approach summary |
| ------ | -------------------------- |
| **DTP-ViT** (ours!) | a single boundary predictor using **Gumbel-Sigmoid** |
| DynamicViT (2021) | a binary decision mask to **PRUNE** tokens at each transformer layer |
| TokenLearner (2021) | a spatial attention module inserted in ViT to **LEARN** tokens |  
| NativeSegViT (2025) | kmeans-like clustering to dynamically **GROUP** tokens repeatedly |

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

## Performance Metrics

Top-1 Acc (%) and Top-5 Acc (%)

Important observation: **STILL NEED MUCH MORE DATA** - ViT-B-32 after 50 epochs:
```
2025-06-08,06:13:34 | INFO | Eval Epoch: 50 imagenet-zeroshot-val-top1: 0.0250	imagenet-zeroshot-val-top5: 0.0813
```

## Efficiency Metrics

1. GFLOPs: a different script (adapted from **DynamicViT**), NOT during training
     1. a **pretrained** ViT-B-32 is used to compute FLOPs for ViT-B-32
     2. important Adaptations from DynamicViT:
          [FLOP measurement](https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py)
          [simulating artificial bounddaries for DynamicViT](https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py)

2. GPU memory and training step time are averaged for each epoch.
     1. memory: torch.cuda.max_memory_allocated()
     2. training step time: **already built-in** by CLIP!

**IMPORTANT** observation:
1. the first epoch takes **longer** time than the following epochs: example - 1.428s, 0.641s, 0.608s, 0.609s, 0.615s...

## LAION-400M dataset preparation - training from scratch

| preparation strategy | ideal? |
| -------- | ------ |
| stream **HuggingFace** dataset for local download with single process | ❌: slow, impossible to use |
| stream **HuggingFace** dataset for local download with multi-processing | ⚠️: still suboptimal, I/O bottleneck |
| stream arbitrary number of samples **"on the fly"** during training | 🤡: NASTY, slow down training by too much |
| download parquet metadata, then use **img2dataset** | ✅: the best solution so far |

Current status: **"10M"** samples (success rate = 65%, so ***effectively*** 6.5M samples 🥲) - we have **1000** shards in total

## DTP-ViT results - training from scratch

### 1M subset of LAION-400M - # epochs = 2, batch size = 512

| model | GFLOPs (fvcore) | resolution | patch size | Top-1 Acc (%) | Top-5 Acc (%) | avg GPU memory (GB) | avg training step time (s) |
| ------- | ----- | --------------- | ---------- | ---------- | ---------------- | ------------- | ---------- |
| **ViT-B-32** | 2.96 | 224 | 32 | **1.20%** | **4.55%** | **20.1** | 0.837 |
| **2x compression** | 3.01 | 224 | 32 | 1.03% | 4.27% | 22.0 | 0.699 |
| **4x compression** | 2.32 | 224 | 32 | 0.99% | 4.35% | 21.8 | 0.709 |
| **10x compression** | 1.86 | 224 | 32 | 1.11% | 4.34% | 21.9 | **0.696** |
| **2x, no upsampling** | 2.67 | 224 | 32 | 0.89% | 4.21% | 20.4 | 0.790 |
| **4x, no upsampling** | 1.82 | 224 | 32 | 1.00% | 3.95% | 20.4 | 0.788 |
| **10x, no upsampling** | **1.25** | 224 | 32 | 0.97% | 4.01% | **20.2** | 0.798 |

## DTP-ViT results - finetuning on ImageNet

### zero-shot performance of pretrained CLIPs

| pretrained vision encoder | corresponding dataset | zero-shot dataset | zero-shot top-1 | zero-shot top-5 |
| ------------------------- | --------------------- | ----------------- | --------------- | --------------- |
| RN50x16 | openai | ImageNet (2012), 50k val | 70.14% | 92.41% |
| ViT-B-32 (88M) | laion2b_s34b_b79k | ImageNet (2012), 50k val | 66.53% | 89.89% |
| ViT-B-16 (86M) | laion2b_s34b_b88k | ImageNet (2012), 50k val | 70.21% | 91.76% |
| ViT-L-14 (307M) | laion2b_s32b_b82k | ImageNet (2012), 50k val | 75.26% | 94.25% |


### finetune CLIP-pretrained ViTs on ImageNet-1K (1.28M images)

| model | dataset pretrained on | freeze the backbone? | batch size | epoch | zero-shot (as reference) |classification accuracy |
| ----- | --------------------- | ----------------- | -------- |
| ViT-B-32 | laion2b_s34b_b79k | yes | 512 | 1 | 66.53% | 👍🏻**67.73%** | 
| ViT-B-32 | laion2b_s34b_b79k | finetune all | 512 | 1 | 66.53% | 👎🏻50.02% |
| 10x compression | load weights from ViT-B-32 | entirely | 128 | 1 | 🤡1.46% |
