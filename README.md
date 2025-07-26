# 💧 DRIP: **D**ynamic token **R**eduction v**I**sion transformer via **P**ooling for efficient multimodal learning

## DRIP v.s. existing work

| design | approach summary |
| ------ | -------------------------- |
| **DRIP** (ours!) | a single boundary predictor using **Gumbel-Sigmoid** |
| DynamicViT (2021) | a binary decision mask to **PRUNE** tokens at each transformer layer |
| TokenLearner (2021) | a spatial attention module inserted in ViT to **LEARN** tokens |  
| NativeSegViT (2025) | kmeans-like clustering to dynamically **GROUP** tokens repeatedly |

According to DTP paper, both **Gumbel-Sigmoid** and **Entropy-Spike** are very suitable to adapt to other modalities!

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
| DRIP-10x-32, 2+24 | 1.86✅ | 224 | 32 | 10 | **running** | running | **running** | **running** |
| ViT-B-16 | 11.33 | 224 | 32 | 10 | **33.88%** | 60.81% | **43.9** | **0.756** |
| DRIP-2x-16, 2+10 | 10.22 | 224 | 16 | 10 | **30.59%** | 57.11% | **43.0** | **0.706** |
| DRIP-4x-16, 2+10 | 6.62 | 224 | 16 | 10 | **28.25%** | 53.95% | **32.2** | **0.570** |
| DRIP-10x-16, 2+10 | 4.53 | 224 | 16 | 10 | **26.36%** | 50.79% | **26.3** | **0.515** |

## pooling across tokens

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
- [ ] average pooling **excluding** CLS/first token
  - [ ] DRIP-2x-32, 10 epochs of 28M: running now!
  - [ ] DRIP-4x-32, 10 epochs of 28M: running now!
- [ ] CLS/first token pooling
  - [ ] DRIP-2x-32, 10 epochs of 28M: running now!
  - [ ] DRIP-4x-32, 10 epochs of 28M: submitted!
- [ ] last token pooling
  - [ ] DRIP-2x-32, 10 epochs of 28M: 
  - [ ] DRIP-4x-32, 10 epochs of 28M:

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

Note: CLIP people trained their models for **32** epochs instead of 10.

## TASK 2 - ImageNet Classification Finetuning

### Performance Metrics

Classification accuracy on ImageNet

### train ViTs on ImageNet-1K (1.28M images)

| model | dataset pretrained on | zero-shot | freeze the backbone? | epoch | classification accuracy |
| ----- | --------------------- | -------------------- | ---------- | ----- | ----------------------- |
| <tr><td colspan="6" align="center"> pretrained ViT </td></tr> |
| ViT-B-32 | laion2b_s34b_b79k | 66.53% | yes | 30 | 🟢76.81% |
| ViT-B-32 | laion2b_s34b_b79k | 66.53% | no | 30 | 🟠60.98% |
| <tr><td colspan="6" align="center"> pretrained DRIP </td></tr> |
| DRIP-2X-16 | 3 epochs of 280M LAION | **36.71%** | no | 100 | **42.30%** |


## TASK 3 - Visual Instruction Tuning (LLaVA)

### Pretraining - Feature Alignment with **558K** samples 
     
1. image-caption data (LAION-CC-SBU with BLIP captions) -> conversation data
2. objective: image + question -> response
3. both visual encoder and LLM are frozen, only train the projectors

### Visual Instruction Tuning

dataset (LLaVA-Instruct-158K): 158K samples (58K conversations, 23K detailed description, 77K in complex reasoning)

finetuning on (i) multimodal chatbot using LLaVA-Instruct-158K for **3 epochs** (ii) Science QA benchmark; **only visual encoder is frozen**, update LLM and the projectors

### DRIP Integration debugging list

- pretraining
  - [x] precision matching: enforce float32 since it's the precision used in my DRIP checkpoint
  - [x] vision projector: ensure `embed_dim` alignment between the encoder and projector
  - [x] forward pass, but **ONLY imtermediates**: no average pooling across tokens!
- finetuning
  - [x] fixed Cuda OOM issue with finetuning
  - [x] float16 (HALF) instead of float32 (FLOAT)
  - [x] LoRA enabled
  - [ ] convert all files as .jpg in OCR-VQA [fixing]
  ```java
  [rank3]: FileNotFoundError: [Errno 2] No such file or directory: '/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/ocr_vqa/images/689852649.jpg'
  ```
  - [ ] cannot decrease scale anymore? 
  ```java
  [rank0]: Exception: Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.
  ```

### LLaVa Evaluation

benchmarks:

- [ ] ScienceQA (LLM-free)
  - [x] evaluation setup
     - [x] fix dependency issues with transformers==4.41.2
  - [ ] DRIP-2X-16 (**36.71%** zero-shot)

| model details | Accuracy | IMG-Accuracy |
| --------------------- | -------- | ------------ |
| llava-v1.5-13b from HuggingFace | 68.43% | 70.45% |
| DRIP-2X-16 (1 epoch pretrain + 1 epoch finetune) | N/A | N/A |


- [ ] LLaVA-Bench-in-the-Wild (LLM-judge)
  - [x] evaluation setup
  - [ ] DRIP-2X-16 (**36.71%** zero-shot)

| model details | LLM-judge (A, C, D, R) | Rule-Based (A, C, D, R) | Overall (A, C, D, R) |   
| ------------- | ---------------------- | ----------------------- | -------------------- |
| llava-v1.5-13b | (88.7%, 91.8%, 88.0%, 87.1%) | (64.8%, 57.1%, 55.3%, 74.5%) | (73.0%, 62.2%, 62.9%, 85.4%) |


Note: (A, C, D, R) = (Average, Conversation, Detail description, Complex reasoning)

