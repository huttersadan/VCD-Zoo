# VCD-Zoo

A unified toolkit for contrastive decoding-based LVLM hallucination mitigation methods.

This repository is the runnable entry point for the organized VCD experiments.

It lets you choose:

- method: `original`, `vcd`, `avisc`, `agla`
- model: `llava`, `instructblip2` / `blip2`, `internvl`
- benchmark: `chair`, `pope`, `mme`

The launcher calls the local runner scripts inside this folder and writes
outputs under `./outputs`. It uses plain `python` execution only, with one
child process per selected experiment.

## Environment

The likely environment is:

```bash
conda activate llava
cd /data/dtt/projects/VCD/unified_experiments
```

Check before running:

```bash
python check_env.py
```

`run_experiment.py` sets `HF_ENDPOINT=https://hf-mirror.com` by default for
HuggingFace downloads. If you need another mirror, export `HF_ENDPOINT` before
running and the launcher will keep your value.

## Dry Run

Preview the command without loading models:

```bash
python run_experiment.py --method vcd --model llava --benchmark pope --dry-run
```

Preview a 5-sample test:

```bash
python run_experiment.py --method vcd --model llava --benchmark pope --limit-samples 5 --dry-run
```

## Run Examples

Run one CHAIR experiment:

```bash
python run_experiment.py \
  --method vcd \
  --model llava \
  --benchmark chair \
  --cuda-visible-devices 0
```

Run one POPE subset:

```bash
python run_experiment.py \
  --method avisc \
  --model internvl \
  --benchmark pope \
  --type_dataset coco \
  --type_question random \
  --cuda-visible-devices 0
```

Run all POPE subsets:

```bash
python run_experiment.py \
  --method original \
  --model instructblip2 \
  --benchmark pope \
  --all \
  --cuda-visible-devices 0
```

Run one MME category:

```bash
python run_experiment.py \
  --method vcd \
  --model llava \
  --benchmark mme \
  --mme_name existence \
  --cuda-visible-devices 0
```

Run all wired MME categories:

```bash
python run_experiment.py \
  --method avisc \
  --model blip2 \
  --benchmark mme \
  --all \
  --cuda-visible-devices 0
```

Run the first 5 samples for every wired method/model/benchmark combination:

```bash
./scripts/run_all_combinations.sh 5 0
```

Run the 32 runnable high-level combinations in 4 parallel groups of 8 jobs:

```bash
./scripts/run_group.sh 1 5
./scripts/run_group.sh 2 5
./scripts/run_group.sh 3 5
./scripts/run_group.sh 4 5
```

Run the first 5 samples for the full benchmark sweep, including all POPE and
MME subsets:

```bash
./scripts/run_full_sweep.sh 5 0
```

## Coverage

| Benchmark | llava | instructblip2/blip2 | internvl |
| --- | --- | --- | --- |
| chair | original, VCD, AVISC, AGLA | original, VCD, AVISC, AGLA | original, VCD, AVISC, AGLA |
| pope | original, VCD, AVISC, AGLA | original, VCD, AVISC, AGLA | original, VCD, AVISC, AGLA |
| mme | original, VCD, AVISC, AGLA | original, VCD, AVISC, AGLA | not wired yet |

`internvl + mme` is not connected because the current runner code does not have
that implementation yet.

There are 36 theoretical method/model/benchmark combinations:
`4 methods * 3 models * 3 benchmarks`. The current runnable set is 32 because
`internvl + mme` is missing for all 4 methods.

If POPE and MME internal subsets are counted as separate jobs, a full sweep is:

- CHAIR: `4 methods * 3 models = 12` jobs
- POPE: `4 methods * 3 models * 3 datasets * 3 question types = 108` jobs
- MME: `4 methods * 2 wired models * 5 categories = 40` jobs

That is `160` runnable jobs in total.

## Result Paths

Results are saved under `./outputs`:

- CHAIR: `outputs/chair_output/<model>/<method>/captions.jsonl`
- POPE: `outputs/pope_output/<model>/<method>/<dataset>_<question>/results.txt`
- MME: `outputs/mme_output/<model>/<method>/<category>.txt`

POPE and MME also save per-sample model responses:

- POPE responses: `outputs/pope_output/<model>/<method>/<dataset>_<question>/responses.jsonl`
- MME responses: `outputs/mme_output/<model>/<method>/responses/<category>.jsonl`

## Single-Process Behavior

`--cuda-visible-devices` only sets the `CUDA_VISIBLE_DEVICES` environment
variable for the child process. For normal single-GPU runs, pass one GPU id:

```bash
python run_experiment.py --method vcd --model llava --benchmark chair --cuda-visible-devices 0
```
