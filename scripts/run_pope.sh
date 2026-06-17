#!/usr/bin/env bash
set -euo pipefail

METHOD=${1:-vcd}
MODEL=${2:-llava}
TYPE_DATASET=${3:-coco}
TYPE_QUESTION=${4:-popular}
GPUS=${5:-0}
LIMIT=${6:-}

cd "$(dirname "$0")/.."
cmd=(python run_experiment.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --benchmark pope \
  --type_dataset "$TYPE_DATASET" \
  --type_question "$TYPE_QUESTION" \
  --cuda-visible-devices "$GPUS")

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit-samples "$LIMIT")
fi

"${cmd[@]}"
