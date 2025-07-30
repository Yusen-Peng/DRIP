# 💧 DRIP: **D**ynamic token **R**eduction v**I**sion transformer via **P**ooling for efficient multimodal learning

## DRIP Architecture

![alt text](/docs/DRIP.png)

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

![alt text](unit_visualization/boundary_visualization_0_2x_32.png)
![alt text](unit_visualization/boundary_visualization_0_4x_32.png)
![alt text](unit_visualization/boundary_visualization_0_10x_32.png)

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
| ViT-B-32 | **2.96** | 224 | 32 | 10 | **28.77%** | 54.34% | 20.1 | 0.429 |
| DRIP-2x-32, 2+10 | 2.69 | 224 | 32 | 10 | **25.72%** | 49.95% | **18.4** | **0.412** |
| DRIP-2x-32, 2+11 | 2.87⚠️ | 224 | 32 | 10 | **26.13%** | 50.91% | **18.9** | **0.398** |
| DRIP-4x-32, 2+10 | 1.83 | 224 | 32 | 10 | **24.24%** | 47.82% | **16.3** | **0.378** |
| DRIP-4x-32, 2+12 | 2.03✅ | 224 | 32 | 10 | **23.87%** | 47.23% | **16.9** | **0.383** |
| DRIP-4x-32, 2+16 | 2.43✅ | 224 | 32 | 10 | **24.54%** | 48.70% | **18.5** | **0.402** |
| DRIP-4x-32, 2+18 | 2.63✅ | 224 | 32 | 10 | **24.67%** | 48.75% | **19.2** | **0.405** |
| DRIP-4x-32, 2+20 | 2.83⚠️ | 224 | 32 | 10 | **25.67%** | 49.81% | **19.8** | **0.398** |
| DRIP-10x-32, 2+10 | 1.26 | 224 | 32 | 10 | **21.70%** | 44.30% | **15.0** | **0.365** |
| DRIP-10x-32, 2+24 | 1.86✅ | 224 | 32 | 10 | **22.52%** | 45.27% | **18.5** | **0.391** |
| ViT-B-16 | 11.33 | 224 | 32 | 10 | **33.88%** | 60.81% | **43.9** | **0.756** |
| DRIP-2x-16, 2+10 | 10.22 | 224 | 16 | 10 | **30.59%** | 57.11% | **43.0** | **0.706** |
| DRIP-4x-16, 2+10 | 6.62 | 224 | 16 | 10 | **28.25%** | 53.95% | **32.2** | **0.570** |
| DRIP-10x-16, 2+10 | 4.53 | 224 | 16 | 10 | **26.36%** | 50.79% | **26.3** | **0.515** |

### pooling across tokens

ViT pooling implementation (average pooling excluding CLS/first token OR CLS/first token pooling):
```py
def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
     if self.pool_type == 'avg':
          pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
     elif self.pool_type == 'tok':
          pooled, tokens = x[:, 0], x[:, 1:]
     else:
          pooled = tokens = x

     return pooled, tokens
```

but the default one is CLS/first token pooling:

```py
pool_type: str = 'tok'
```

Our DRIP original pooling implementation (average pooling **including** CLS/first token):
```py
# downsample patch tokens
patch_tokens = downsample(hard_boundaries, patch_tokens, self.null_group)
patch_tokens = patch_tokens[1:]  # remove null group at index 0 → [S, B, D]

# reattach CLS token
x = torch.cat([cls_token, patch_tokens], dim=0)     # [1 + S, B, D]

x = self.norm(x)                                     # [1 + S, B, D]
x = self.shorten_blocks(x)                           # [1 + S, B, D]

x = x.transpose(0, 1)                               # [B, 1 + S, D]
x = self.norm(x)                                    # [B, 1 + S, D]
x = x.mean(dim=1)                                   # [B, D]
```

alternatives:
- [x] average pooling **excluding** CLS/first token
  - [x] DRIP-2x-32, 10 epochs of 28M: 25.96% (originally 25.72%)
  - [x] DRIP-4x-32, 10 epochs of 28M: 23.23% (originally 24.24%)
- [x] CLS/first token pooling
  - [x] DRIP-2x-32, 10 epochs of 28M: 24.17% (originally 25.72%)
  - [x] DRIP-4x-32, 10 epochs of 28M: 23.34% (originally 24.24%)
- [x] last token pooling (we hope it's cumulative!)
  - [x] DRIP-2x-32, 10 epochs of 28M: 25.25% (originally 25.72%)
  - [x] DRIP-4x-32, 10 epochs of 28M: 24.99% (originally 24.24%)

### High LR: gradient explodes!

Higher learning rate is subject to gradient explosion (for Sigmoid function specifically):

```cpp
ValueError: Expected parameter probs (Tensor of shape (512, 49)) of distribution LogitRelaxedBernoulli(probs: torch.Size([512, 49])) to satisfy the constraint Interval(lower_bound=0.0, upper_bound=1.0), but found invalid values:
tensor([[nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        ...,
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan],
        [nan, nan, nan,  ..., nan, nan, nan]], device='cuda:0',
       dtype=torch.float16, grad_fn=<SigmoidBackward0>)
```

my proposed fix(es):

- [x] gradient clipping
     ```bash
     "--grad-clip-norm", "1.0", # gradient clipping
     ```
- [x] logit clamping
     ```python
     boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
     # before sigmoid
     boundary_logits = torch.clamp(boundary_logits, min=-20.0, max=20.0)
     boundary_probs = torch.sigmoid(boundary_logits)
     # after sigmoid
     boundary_probs = torch.clamp(boundary_probs, min=1e-4, max=1 - 1e-4)
     ```

### Resume from the last checkpoint

Successfully load the **last checkpoint** and the **optimizer state**, too:

```csharp
"--resume", "latest", # resume from the latest checkpoints
"--checkpoint-path", "logs/DRIP-2X-16/checkpoints", # the path to save checkpoints
```

but... the learning rate will always be re-initialized!

```python
# create scheduler if train
scheduler = None
if 'train' in data and optimizer is not None:
     total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
     if args.lr_scheduler == "cosine":
          scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
     elif args.lr_scheduler == "const":
          scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
```

my proposed fix:

```python
# proposed fix: override args.lr with the optimizer's current lr if resuming
if args.resume is not None and optimizer is not None:
     args.lr = optimizer.param_groups[0]['lr']
     print("🧠" * 20)
     print(f"Overriding args.lr with optimizer's current lr: {args.lr}", flush=True)
```

final results 🎉:

```csharp
2025-07-29,22:52:07 | INFO | => resuming checkpoint 'logs/DRIP-2X-16/checkpoints/epoch_3.pt' (epoch 3)
🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠🧠
Overriding args.lr with optimizer's current lr: 7.939449900550195e-05
```


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

## TASK 2 - ImageNet Finetuning

### train ViTs on ImageNet-1K (1.28M images)

| model | dataset pretrained on | zero-shot | freeze the backbone? | epoch | classification accuracy |
| ----- | --------------------- | -------------------- | ---------- | ----- | ----------------------- |
| <tr><td colspan="6" align="center"> pretrained ViT </td></tr> |
| ViT-B-32 | laion2b_s34b_b79k | 66.53% | yes | 30 | 🟢76.81% |
| ViT-B-32 | laion2b_s34b_b79k | 66.53% | no | 30 | 🟠60.98% |
| <tr><td colspan="6" align="center"> pretrained DRIP </td></tr> |
| DRIP-2X-16 | 3 epochs of 280M LAION | **36.71%** | no | 100 | **🟢42.30%** |

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

