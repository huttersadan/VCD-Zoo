#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:-vcd}
MODEL=${2:-llava}
GPUS=${3:-0}
LIMIT=${4:-}

cd "$(dirname "$0")/.."
cmd=(python run_experiment.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --benchmark mme \
  --all \
  --cuda-visible-devices "$GPUS")

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit-samples "$LIMIT")
fi

"${cmd[@]}"
