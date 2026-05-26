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



# Reasoning (2): GQA
sbatch scripts/task3/eval/eval_GQA.sh
# MMMU
sbatch scripts/task3/eval/eval_MMMU.sh




# OCR (2): TextVQA
sbatch scripts/task3/eval/eval_textVQA.sh
# OCRBench
sbatch scripts/task3/eval/eval_ocrbench.sh



# Hallucination (1): POPE
sbatch scripts/task3/eval/eval_POPE.sh




# Free Response (2): LLaVA-in-the-wild
sbatch scripts/task3/eval/eval_in_the_wild.sh
# MM-Vet
# need to submit the result json file to:
# https://huggingface.co/spaces/whyu/MM-Vet_Evaluator
sbatch scripts/task3/eval/eval_MMVet.sh

