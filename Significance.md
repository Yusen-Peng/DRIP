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

```bash
# 💚💚💚 convert files first
python src/convert_gqa_for_eval.py \
  --src /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced/LLaVA_7B_Fixed_4x_finetune_train_full.jsonl \
  --dst /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced/LLaVA_7B_Fixed_4x_finetune_train_full_predictions.json

python src/convert_gqa_for_eval.py \
  --src /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced/LLaVA_7B_DRIP_4x_finetune_train_full.jsonl \
  --dst /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced/LLaVA_7B_DRIP_4x_finetune_train_full_predictions.json

# now we can actually do the test
python src/mcnemar_test/GQA_test.py \
  --questions-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/data/questions1/testdev_balanced_questions.json \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced/LLaVA_7B_Fixed_4x_finetune_train_full_predictions.json \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced/LLaVA_7B_DRIP_4x_finetune_train_full_predictions.json
```

## MMMU

```bash
python src/mcnemar_test/MMMU_test.py \
  --answer-path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answer_key/answer_dict_val.json \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/LLaVA_7B_Fixed_4x_finetune_train_full.json \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/LLaVA_7B_DRIP_4x_finetune_train_full.json
```

## TextVQA

```bash
python src/mcnemar_test/TextVQA_test.py \
  --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/TextVQA_0.5.1_val.json \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/LLaVA_7B_Fixed_4x_finetune_train_full.jsonl \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/LLaVA_7B_DRIP_4x_finetune_train_full.jsonl
```

## OCRBench

```bash
python src/mcnemar_test/OCRBench_test.py \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results/LLaVA_7B_Fixed_4x_finetune_train_full.json \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results/LLaVA_7B_DRIP_4x_finetune_train_full.json
```

## OCRBench v2

```bash
python src/mcnemar_test/OCRBenchv2_test.py \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores/LLaVA_7B_Fixed_4x_finetune_train_full_scores.json \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores/LLaVA_7B_DRIP_4x_finetune_train_full_scores.json
```

## DocVQA

```bash
python src/mcnemar_test/DocVQA_test.py \
  --baseline-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/LLaVA_7B_Fixed_4x_finetune_train_full_eval.json \
  --method-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/LLaVA_7B_DRIP_4x_finetune_train_full_eval.json
```


## ChartQAPro


```bash

```

