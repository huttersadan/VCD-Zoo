#!/usr/bin/env bash
set -euo pipefail

LIMIT=${1:-5}
LOG_DIR=${2:-logs/remaining}

cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

declare -a GPUS=(0 1 2 3 4 5 6)

declare -a JOBS=(
  "original llava pope --type_dataset coco --type_question popular"
  "original llava mme --mme_name existence"
  "original blip2 pope --type_dataset coco --type_question popular"
  "original blip2 mme --mme_name existence"
  "original internvl chair"
  "original internvl pope --type_dataset coco --type_question popular"
  "vcd llava chair"
  "vcd llava mme --mme_name existence"
  "vcd blip2 chair"
  "vcd blip2 pope --type_dataset coco --type_question popular"
  "vcd internvl chair"
  "vcd internvl pope --type_dataset coco --type_question popular"
  "avisc llava chair"
  "avisc llava pope --type_dataset coco --type_question popular"
  "avisc llava mme --mme_name existence"
  "avisc blip2 chair"
  "avisc blip2 pope --type_dataset coco --type_question popular"
  "avisc blip2 mme --mme_name existence"
  "avisc internvl chair"
  "avisc internvl pope --type_dataset coco --type_question popular"
  "agla llava chair"
  "agla llava pope --type_dataset coco --type_question popular"
  "agla llava mme --mme_name existence"
  "agla blip2 chair"
  "agla blip2 pope --type_dataset coco --type_question popular"
  "agla blip2 mme --mme_name existence"
  "agla internvl chair"
  "agla internvl pope --type_dataset coco --type_question popular"
)

pids=()
names=()

cleanup() {
  if ((${#pids[@]} > 0)); then
    echo "Stopping running jobs: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

wait_batch() {
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  pids=()
  names=()
  return "$failed"
}

failed=0
slot=0
for job in "${JOBS[@]}"; do
  read -r method model benchmark extra <<<"$job"
  gpu=${GPUS[$slot]}
  name="gpu${gpu}_${method}_${model}_${benchmark}"
  log_file="$LOG_DIR/${name}.log"

  echo "start $name -> $log_file"
  python run_experiment.py \
    --method "$method" \
    --model "$model" \
    --benchmark "$benchmark" \
    --limit-samples "$LIMIT" \
    --cuda-visible-devices "$gpu" \
    ${extra:-} \
    >"$log_file" 2>&1 &

  pid=$!
  pids+=("$pid")
  names+=("$name")
  echo "$pid $name $log_file" >>"$LOG_DIR/pids.txt"

  slot=$((slot + 1))
  if ((slot == ${#GPUS[@]})); then
    if ! wait_batch; then
      failed=1
    fi
    slot=0
  fi
done

if ((${#pids[@]} > 0)); then
  if ! wait_batch; then
    failed=1
  fi
fi

trap - INT TERM

if [[ "$failed" -eq 0 ]]; then
  echo "All remaining jobs finished successfully."
else
  echo "At least one remaining job failed. Check logs in $LOG_DIR."
fi

exit "$failed"
