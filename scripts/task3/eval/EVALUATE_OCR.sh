#!/bin/bash
# OCR (5): TextVQA
sbatch scripts/task3/eval/eval_textVQA.sh
# OCRBench
sbatch scripts/task3/eval/eval_ocrbench.sh
# OCRBench-v2
sbatch scripts/task3/eval/eval_ocrbenchv2.sh
# DocVQA
sbatch scripts/task3/eval/eval_DocVQA.sh
# ChartQAPro
sbatch scripts/task3/eval/eval_ChartQAPro.sh
