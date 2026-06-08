# Significance Test - McNemar's Test

## SQA

```bash
python src/mcnemar_test/SQA_test.py \
    --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/LLaVA_7B_Fixed_4x_finetune_train_full_output.jsonl \
    --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/LLaVA_7B_DRIP_4x_finetune_train_full_output.jsonl
```

## MME

```bash
python src/mcnemar_test/MME_test.py \
  --baseline-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/eval_tool/answers/LLaVA_7B_Fixed_4x_finetune_train_full \
  --method-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/eval_tool/answers/LLaVA_7B_DRIP_4x_finetune_train_full \
  --eval-type Perception
```

## MMBench

```bash
python src/mcnemar_test/MMBench_test.py \
  --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/mmbench_dev_20230712.tsv \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712/LLaVA_7B_Fixed_4x_finetune_train_full.jsonl \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712/LLaVA_7B_DRIP_4x_finetune_train_full.jsonl
```

## GQA
