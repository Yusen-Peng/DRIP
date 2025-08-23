torchrun --nproc_per_node=4 src/task1_newcodebase.py \
    --model vit_b_16 --epochs 300 --batch-size 256 --opt adamw --lr 0.003 --wd 0.3 \
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 \
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra \
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema