# DRIP💧: **D**ynamic token **R**eduction v**I**sion transformer via **P**ooling for efficient multimodal learning

## DRIP Architecture

![alt text](/docs/DRIP.png)

## Efficiency Metrics

1. GFLOPs: a different script (adapted from **DynamicViT**), NOT during training
     1. a **pretrained** ViT-B-32 is used to compute FLOPs for ViT-B-32
     2. important Adaptations from DynamicViT:
          [FLOP measurement](https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py)
          [simulating artificial boundaries for DynamicViT](https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py)

GFLOP refinement: change from interval-based boundary simulation to proportion-based boundary simulation in order to push boundary rate beyond 50%:

interval-based approach:

```python
interval = max(1, int(1 / self.compression_rate))
hard_boundaries = torch.zeros(B, L, device=x.device)
hard_boundaries[:, ::interval] = 1
soft_boundaries = hard_boundaries.clone()
```
proportion-based approach:

```python
L = patch_tokens.shape[0]
num_tokens_to_keep = max(1, int(L * self.compression_rate))
indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
hard_boundaries = torch.zeros(B, L, device=x.device)
hard_boundaries[:, indices] = 1
```

2. GPU memory and training step time are averaged for each epoch
     1. memory: torch.cuda.max_memory_allocated()
     2. training step time

## TASK 1 - Contrastive Pretraining (CLIP)

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

### LAION-2B subset (26M samples) results

| model | GFLOPs (fvcore) | resolution | patch size | #epochs | Top-1 Acc (%) | Top-5 Acc (%) | avg GPU memory (GB) | avg training step time (s) |
| ------- | ----- | --------------- | ---------- | -------- | ---------- | ---------------- | ------------- | ---------- |
| ViT-B-32 | **2.95🔥** | 224 | 32 | 10 | **28.77%** | 54.34% | 20.1 | 0.429 |
| DRIP-2x-32, 2+10 | **2.62** | 224 | 32 | 10 | **25.72%** | 49.95% | **18.4** | **0.412** |
| DRIP-2x-32, 2+11 | 2.8 | 224 | 32 | 10 | **26.13%** | 50.91% | **18.9** | **0.398** |
| DRIP-4x-32, 2+10 | **1.76** | 224 | 32 | 10 | **24.24%** | 47.82% | **16.3** | **0.378** |
| DRIP-4x-32, 2+20 | 2.68 | 224 | 32 | 10 | **25.67%** | 49.81% | **19.8** | **0.398** |
| DRIP-10x-32, 2+10 | **1.19** | 224 | 32 | 10 | **21.70%** | 44.30% | **15.0** | **0.365** |
| DRIP-10x-32, 2+24 | 1.69 | 224 | 32 | 10 | **22.52%** | 45.27% | **18.5** | **0.391** |
| ViT-B-16 | **11.29🌝** | 224 | 32 | 10 | **33.88%** | 60.81% | **43.9** | **0.756** |
| DRIP-2x-16, 2+10 | 10.22 | 224 | 16 | 10 | **30.59%** | 57.11% | **43.0** | **0.706** |
| DRIP-4x-16, 2+10 | 6.62 | 224 | 16 | 10 | **28.25%** | 53.95% | **32.2** | **0.570** |
| DRIP-10x-16, 2+10 | 4.46 | 224 | 16 | 10 | **26.36%** | 50.79% | **26.3** | **0.515** |
| <tr><td colspan="9" align="center"> higher boundary rate, but fewer layers </td></tr> |
| DRIP-1.6x-32 (60%), 2+9 | **2.76🔥** | 224 | 32 | 10 | **24.32%** | 48.75% | **10.4 (I cut the batch size by half)** | **0.191** |
| <tr><td colspan="9" align="center"> delay pooling </td></tr> |
| DRIP-2x-32, 3+9 | **2.8🔥** | 224 | 32 | 10 | **25.09%** | 49.66% | **10.3 (I cut the batch size by half)** | **0.194** |


### fix mean pooling by not pooling padded tokens

For this set of experiments: 

```bash
learning_rate = 5e-5
batch_size = 512
GPU = 4
```

| model details | GFLOPs | epochs | top-1 zero-shot | top-5 zero-shot |
| ----- | ------ | ------ | ----- | ----- |
| ViT-B-16 | 11.29 | 6 | **27.44%** | **52.77%** |
| 💧DRIP-4X-16, 4+8 | 9.61🔥 | 6 | **26.26%🔥** | **51.64%🔥** |
| 💧DRIP-4X-16, 2+10 | 7.21 | 6 | **23.79%** | **48.11%** |
| 💦H-DRIP-16-50%-50%, 3+3+6 | 9.65 | **5🌝** | **22.88%** | **46.69%** |
| 🫧S-DRIP-16-20%-30%, 2+10 | 7.21 | 6 | **24.51%** | **48.98%** |

![alt text](/NEW_acc_vis_16.png)

Train with exact 10 epochs:

| model details | GFLOPs | epochs | top-1 zero-shot | top-5 zero-shot |
| ----- | ------ | ------ | ----- | ----- |
| ViT-B-16 | 11.29 | **10** | **?** | **?** |
| 💧DRIP-4X-16, 4+8 | 9.61🔥 | **10** | **?** | **?** |
| 💧DRIP-4X-16, 2+10 | 7.21 | **10** | **?** | **?** |




### What else?

existing bug: 
- [ ] not mask **the null token** in the attention mask 

other things to try:
- [ ] extend training epochs and reduce LR slightly
- [ ] add temperature annealing for RelaxedBernoulli and start with a high value

### LAION-280M (178Msamples, 178,918,585) results

| model | GFLOPs (fvcore) | resolution | patch size | #epochs | Top-1 Acc (%) | Top-5 Acc (%) | avg GPU memory (GB) | avg training step time (s) |
| ------- | ----- | --------------- | ---------- | -------- | ---------- | ---------------- | ------------- | ---------- |
| ViT-B-32 | 2.96 | 224 | 32 | 5 | **39.80%** | 68.55% | 20.1 | 0.419 |
| 2x comp | 2.69 | 224 | 32 | 5 | **35.57%** | 63.42% | **18.6** | **0.394** |
| 4x comp | 1.83 | 224 | 32 | 5 | **33.04%** | 59.58% | **16.4** | **0.368** |
| 10x comp | 1.26 | 224 | 32 | 5 | **30.91%** | 57.21% | **15.2** | **0.361** |
| ViT-B-16 | 11.33 | 224 | 32 | 3 | **39.70%** | 68.43% | 43.9 | 0.743 |
| 2x comp | 10.22 | 224 | 16 | 3 | **36.71%** | 64.98% | **43.4** | **0.703** |
| 4x comp | 6.62 | 224 | 16 | 3 | **34.14%** | 61.89% | **32.3** | **0.557** |
| 10x comp | 4.53 | 224 | 16 | 3 | **32.32%** | 59.40% | **26.2** | **0.486** |

## TASK 2 - ImageNet Classification

### train ViTs on ImageNet-1K (1.28M images)

| model | dataset pretrained on | zero-shot | freeze the backbone? | epoch | classification accuracy |
| ----- | --------------------- | -------------------- | ---------- | ----- | ----------------------- |
| <tr><td colspan="6" align="center"> pretrained ViT </td></tr> |
| ViT-B-32 | laion2b_s34b_b79k | 66.53% | yes | 30 | 🟢76.81% |
| ViT-B-32 | laion2b_s34b_b79k | 66.53% | no | 30 | 🟠60.98% |
| <tr><td colspan="6" align="center"> pretrained DRIP </td></tr> |
| DRIP-2X-16 | 3 epochs of 280M LAION | **36.71%** | no | 100 | **🟢42.30%** |
| | <tr><td colspan="6" align="center"> training from scratch </td></tr> |
| ViT-B-16 | N/A | N/A | N/A | 100 | **running!** |
| DRIP-2X-16, 4+8  | N/A | N/A | N/A | 100 | **42.31%** |
| DRIP-2X-16, 2+10 | N/A | N/A | N/A | 100 | **34.88%** |

## TASK 3 - Visual Instruction Tuning (LLaVA)

### LLaVA Benchmark Evaluation

#### ScienceQA (LLM-free)

| model details | Accuracy | IMG-Accuracy |
| --------------------- | -------- | ------------ |
| llava-v1.5-13b from HuggingFace | 68.43% | 70.45% |
| DRIP-2X-16 (**36.71%** zero-shot, 1 epoch pretrain + 1 epoch finetune) | 67.20% | 61.92% |

#### LLaVA-Bench-in-the-Wild (LLM-judge AND rule-based)

| model details | LLM-judge (A, C, D, R) | Rule-Based (A, C, D, R) | Overall (A, C, D, R) |   
| ------------- | ---------------------- | ----------------------- | -------------------- |
| llava-v1.5-13b | (88.7%, 91.8%, 88.0%, 87.1%) | (64.8%, 57.1%, 55.3%, 74.5%) | (73.0%, 62.2%, 62.9%, 85.4%) |
| DRIP-2X-16 (**36.71%** zero-shot, 1 epoch pretrain + 1 epoch finetune) | (93.5%, 94.1%, 98.0%, 90.7%) | (29.8%, 25.3%, 16.0%, 39.8%) | (31.8%, 26.9%, 16.3%, 43.9%) |

Note: (A, C, D, R) = (Average, Conversation, Detail description, Complex reasoning)
