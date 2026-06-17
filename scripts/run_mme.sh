#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:-vcd}
MODEL=${2:-llava}
MME_NAME=${3:-existence}
GPUS=${4:-0}
LIMIT=${5:-}

cd "$(dirname "$0")/.."
cmd=(python run_experiment.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --benchmark mme \
  --mme_name "$MME_NAME" \
  --cuda-visible-devices "$GPUS")

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit-samples "$LIMIT")
fi

"${cmd[@]}"
