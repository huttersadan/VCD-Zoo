#!/usr/bin/env bash
set -euo pipefail

GROUP=${1:?Usage: ./scripts/run_group.sh <1|2|3|4|5> [limit] [log_dir]}
LIMIT=${2:-5}
LOG_DIR=${3:-logs/group${GROUP}}

cd "$(dirname "$0")/.."
mkdir -p "$LOG_DIR"

case "$GROUP" in
  1)
    JOBS=(
      "0 original llava chair"
      "1 original llava pope --type_dataset coco --type_question popular"
      "2 original llava mme --mme_name existence"
      "3 original blip2 chair"
      "4 original blip2 pope --type_dataset coco --type_question popular"
      "5 original blip2 mme --mme_name existence"
      "6 original internvl chair"
      "7 original internvl pope --type_dataset coco --type_question popular"
    )
    ;;
  2)
    JOBS=(
      "0 vcd llava chair"
      "1 vcd llava pope --type_dataset coco --type_question popular"
      "2 vcd llava mme --mme_name existence"
      "3 vcd blip2 chair"
      "4 vcd blip2 pope --type_dataset coco --type_question popular"
      "5 vcd blip2 mme --mme_name existence"
      "6 vcd internvl chair"
      "7 vcd internvl pope --type_dataset coco --type_question popular"
    )
    ;;
  3)
    JOBS=(
      "0 avisc llava chair"
      "1 avisc llava pope --type_dataset coco --type_question popular"
      "2 avisc llava mme --mme_name existence"
      "3 avisc blip2 chair"
      "4 avisc blip2 pope --type_dataset coco --type_question popular"
      "5 avisc blip2 mme --mme_name existence"
      "6 avisc internvl chair"
      "7 avisc internvl pope --type_dataset coco --type_question popular"
    )
    ;;
  4)
    JOBS=(
      "0 agla llava chair"
      "1 agla llava pope --type_dataset coco --type_question popular"
      "2 agla llava mme --mme_name existence"
      "3 agla blip2 chair"
      "4 agla blip2 pope --type_dataset coco --type_question popular"
      "5 agla blip2 mme --mme_name existence"
      "6 agla internvl chair"
      "7 agla internvl pope --type_dataset coco --type_question popular"
    )
    ;;
  5)
    JOBS=(
      "0 original internvl mme --mme_name existence"
      "1 vcd internvl mme --mme_name existence"
      "2 avisc internvl mme --mme_name existence"
      "3 agla internvl mme --mme_name existence"
    )
    ;;
  *)
    echo "GROUP must be 1, 2, 3, 4, or 5" >&2
    exit 2
    ;;
esac

pids=()

cleanup() {
  if ((${#pids[@]} > 0)); then
    echo "Stopping running jobs: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for job in "${JOBS[@]}"; do
  read -r gpu method model benchmark extra <<<"$job"
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
  echo "$pid $name $log_file" >>"$LOG_DIR/pids.txt"
done

echo
echo "Started group $GROUP with ${#pids[@]} jobs."
echo "PID file: $LOG_DIR/pids.txt"
echo "Watch logs with: tail -f $LOG_DIR/*.log"
echo

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

trap - INT TERM

if [[ "$failed" -eq 0 ]]; then
  echo "Group $GROUP finished successfully."
else
  echo "At least one job in group $GROUP failed. Check logs in $LOG_DIR."
fi

exit "$failed"
