# LLaVA image feature analysis

We conduct multiple analyses on image features: (1) PCA; (2) CLS attention; (3) token cosine dissimilarity.

## PCA and CLS attention

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:15:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/feature_visual_LLaVA.py
```

PCA upon 24th-layer features:

![alt text](src/boundary_vis/LLaVA_results/llava_feature_pca_pc1_2x5.png)

CLS attention upon 24th-layer features

(head-mean):

![alt text](src/boundary_vis/LLaVA_results/llava_cls_attn_2x5_mean.png)

(head-max):

![alt text](src/boundary_vis/LLaVA_results/llava_cls_attn_2x5_max.png)


## token cosine similarity

```bash
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:15:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/cossim_visual_LLaVA.py
```
example:

![alt text](src/boundary_vis/LLaVA_results/cosine/pancake_orig_seq_adj_cosine.png)

