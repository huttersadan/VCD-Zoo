#!/usr/bin/env bash
set -euo pipefail

LIMIT=${1:-5}
GPUS=${2:-0}
DRY_RUN=${3:-}

cd "$(dirname "$0")/.."

for method in original vcd avisc agla; do
  for model in llava blip2 internvl; do
    for benchmark in chair pope mme; do
      if [[ "$model" == "internvl" && "$benchmark" == "mme" ]]; then
        echo "skip: method=$method model=$model benchmark=$benchmark is not wired"
        continue
      fi

      cmd=(python run_experiment.py \
        --method "$method" \
        --model "$model" \
        --benchmark "$benchmark" \
        --limit-samples "$LIMIT" \
        --cuda-visible-devices "$GPUS")

      if [[ "$DRY_RUN" == "--dry-run" ]]; then
        cmd+=(--dry-run)
      fi

      "${cmd[@]}"
    done
  done
done
