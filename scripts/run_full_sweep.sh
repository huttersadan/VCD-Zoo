#!/usr/bin/env bash
set -euo pipefail

LIMIT=${1:-5}
GPUS=${2:-0}

cd "$(dirname "$0")/.."

for method in original vcd avisc agla; do
  for model in llava blip2 internvl; do
    python run_experiment.py \
      --method "$method" \
      --model "$model" \
      --benchmark chair \
      --limit-samples "$LIMIT" \
      --cuda-visible-devices "$GPUS"

    python run_experiment.py \
      --method "$method" \
      --model "$model" \
      --benchmark pope \
      --all \
      --limit-samples "$LIMIT" \
      --cuda-visible-devices "$GPUS"

    python run_experiment.py \
      --method "$method" \
      --model "$model" \
      --benchmark mme \
      --all \
      --limit-samples "$LIMIT" \
      --cuda-visible-devices "$GPUS"
  done
done
