#!/usr/bin/env bash
set -euo pipefail

cd /root/SIM-CoT

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/lsp_jepa/data/hf_home"
export HF_DATASETS_CACHE="/root/autodl-tmp/lsp_jepa/data/hf_datasets"

RUN_NAME="steptraj_epoch35_gpt2_gsm8k_aug_full35_20260430"
OUTPUT_ROOT="/root/autodl-tmp/lsp_jepa/runs/lsp_step_trajectory_gpt2_epoch35_gsm8k_aug_full35"
TRAIN_JSONL="/root/autodl-tmp/lsp_jepa/data/gsm8k_aug_train.jsonl"
EVAL_JSONL="/root/autodl-tmp/lsp_jepa/data/gsm8k_aug_test.jsonl"

exec bash lsp_jepa/scripts/launch_step_trajectory_train.sh \
  --max-steps 843360 \
  --output-root "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --device cuda \
  --model-id openai-community/gpt2 \
  --save-every-epoch \
  --keep-last-checkpoints 1 \
  --keep-best-total-loss \
  --eval-every-epoch \
  --eval-json "$EVAL_JSONL" \
  --eval-limit-samples 0 \
  -- \
  --data-path "$TRAIN_JSONL" \
  --dataset-source gsm8k_aug \
  --dataset-id zen-E/GSM8k-Aug \
  --dataset-local-path "$TRAIN_JSONL" \
  --dataset-split train \
  --expected-samples 385531 \
  --min-samples 385531 \
  --hf-endpoint https://hf-mirror.com \
  --effective-rank-every 24096
