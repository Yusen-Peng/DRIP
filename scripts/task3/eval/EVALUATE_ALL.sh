#!/bin/bash

# General VQA (4):
# VQAv2 [🚨LONG🚨]
# need to submit the result json file to:
# https://eval.ai/web/challenges/challenge-page/830
sbatch scripts/task3/eval/eval_VQAv2.sh
# SQA 
sbatch scripts/task3/eval/eval_SQA.sh
# MME
sbatch scripts/task3/eval/eval_MME.sh
# MM-Bench
sbatch scripts/task3/eval/eval_MMBench.sh

# Reasoning (1): GQA
sbatch scripts/task3/eval/eval_GQA.sh

# OCR (1): TextVQA
sbatch scripts/task3/eval/eval_textVQA.sh

# Hallucination (1): POPE
sbatch scripts/task3/eval/eval_POPE.sh

# Free Response (1): LLaVA-in-the-wild
sbatch scripts/task3/eval/eval_in_the_wild.sh
