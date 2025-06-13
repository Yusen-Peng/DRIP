# Literature review

## CLIP (2021)

CLIP architecture: [CLIP](docs/CLIP.png)

CLIP uses the following text encoder and image encoder:

1. text encoder: a standard transformer (attention is all you need)
2. image encoder: ResNet and Vision Transformer (ViT)

WebImageText dataset: **400 million** (image, text) pairs (NOT open sourced)

## Hourglass (2021)

Motivations: vanilla transformers are inefficient

Hourglass architecture: [hourglass](docs/hourglass.png)

Hourglass has **3** transformer blocks: downsample -> process -> upsample

pooling technique: we can merge groups of tokens to reduce sequence length in the intermediate layers (later, these **pooled** representations are up-sampled back to the original length)

## Dynamic Token Pooling (2022)

Motivations: fixed-size pooling is suboptimal: (i) **size** misalignment; (ii) different **degrees** of information (e.g., speaking/silence carry different amount of information)

DPT architecture: [DPT](docs/DPT.png)

neural boundary predictor: it's a sequences of 1s and 0s (1 represents a boundary), implemented as 2-layer MLP; it can effectively replace **off-the-shelf** tokenizers (BPE, Unigram) since they are inconsistent during autoregressive inference.

The neural boundary predictor can be learned upon the following objectives:

1. main objective: **end-to-end** training based on the model perplexity
2. auxiliary objective: tokenization as supervision with Unigram
3. spikes of conditional entropy (reasoning: Empirically, entropy spikes in language models overlap with word boundaries to a significant degree)
4. linguistically inspired segments (we put a boundary after a whitespace character - no need to train)

## MAGNET (2024)

Motivations:

1. current trend: train upon **byte-level** sequences and then do **pooling** to reduce computation and memory
2. however, graidient-based tokenization is tricky across different language scripts

MAGNET architecture: [MAGNET](docs/MAGNET.png)

language-script-specific boundary predictors: ensure modularity/granularity - better than DTP!
