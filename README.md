# VCD-Zoo

A unified toolkit for contrastive decoding-based LVLM hallucination mitigation methods.

This repository provides a single launcher for running `original`, `VCD`, `AvisC`,
and `AGLA` on several LVLM backbones and hallucination benchmarks.

The motivation for this repository is experimental fairness and convenience.
Many hallucination mitigation methods based on contrastive decoding depend on
modifying the decoding implementation inside HuggingFace Transformers. However,
the original releases of these methods often use different environments,
different Transformers versions, and different runner conventions. VCD-Zoo
collects these methods into one runnable codebase so they can be compared under
the same environment and execution interface.

## Supported Settings

Methods:

- `original`: standard greedy search decoding without contrastive decoding.
- `vcd`
- `avisc`
- `agla`

Models:

- `llava`
- `instructblip2` or `blip2`
- `internvl`

Benchmarks:

- `chair`
- `pope`
- `mme`

## Environment

Create a new environment instead of reusing a local experiment environment:

```bash
conda create -n vcd-zoo python=3.8 -y
conda activate vcd-zoo
```

Install PyTorch according to your CUDA version from the official PyTorch
instructions, then install the common Python dependencies used by the runners:

```bash
pip install transformers==4.38.0 accelerate bitsandbytes
pip install pillow tqdm numpy omegaconf einops timm scipy scikit-image opencv-python
pip install diffusers sentencepiece protobuf
```

`transformers==4.38.0` is important. The decoding patches in `vcd_utils/` and
`agla_utils/` monkey-patch generation internals from HuggingFace Transformers.
If you use a newer Transformers version, you may need to adjust those files to
match the updated generation APIs.

## Data And Model Paths

Prepare the model checkpoints and benchmark data before running experiments.
The current runners expect local paths for:

- LLaVA-1.5 checkpoint
- InstructBLIP / BLIP2 checkpoint
- InternVL checkpoint
- MSCOCO images for CHAIR and POPE
- POPE annotations
- MME benchmark data

Update the path constants in the runner files or pass the exposed CLI path
arguments when available, such as `--image_folder` and
`--internvl_model_path`.

## Usage

All experiments are launched through:

```bash
python run_experiment.py \
  --method <original|vcd|avisc|agla> \
  --model <llava|blip2|instructblip2|internvl> \
  --benchmark <chair|pope|mme> \
  --cuda-visible-devices <gpu_id>
```

Examples:

```bash
python run_experiment.py \
  --method vcd \
  --model llava \
  --benchmark chair \
  --cuda-visible-devices 0
```

```bash
python run_experiment.py \
  --method avisc \
  --model blip2 \
  --benchmark pope \
  --type_dataset coco \
  --type_question popular \
  --cuda-visible-devices 0
```

```bash
python run_experiment.py \
  --method agla \
  --model llava \
  --benchmark mme \
  --mme_name existence \
  --cuda-visible-devices 0
```

Useful benchmark options:

- POPE: `--type_dataset <coco|aokvqa|gqa>`
- POPE: `--type_question <random|popular|adversarial>`
- MME: `--mme_name <existence|count|position|OCR|color>`
- POPE/MME: `--all` runs all configured subsets for that benchmark.

For quick debugging, `--limit-samples N` runs only the first `N` samples in the
selected split.

## Result Paths

Results are written under `outputs/`:

- CHAIR captions: `outputs/chair_output/<model>/<method>/captions.jsonl`
- POPE metrics: `outputs/pope_output/<model>/<method>/<dataset>_<question>/results.txt`
- POPE responses: `outputs/pope_output/<model>/<method>/<dataset>_<question>/responses.jsonl`
- MME outputs: `outputs/mme_output/<model>/<method>/<category>.txt`
- MME responses: `outputs/mme_output/<model>/<method>/responses/<category>.jsonl`

`outputs/` and `logs/` are ignored by git.

## Notes

- `--cuda-visible-devices` sets `CUDA_VISIBLE_DEVICES` for the child process.
- AGLA requires the BLIP image-text matching model used to construct the
  augmented visual input.

## Acknowledgements

This codebase integrates and organizes ideas/code from the following works:

- VCD: [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://github.com/DAMO-NLP-SG/VCD)
- AvisC: [Don't Miss the Forest for the Trees: Attentional Vision Calibration for Large Vision Language Models](https://github.com/sangminwoo/AvisC)
- AGLA: [Mitigating Object Hallucinations in Large Vision-Language Models with Assembly of Global and Local Attention](https://github.com/Lackel/AGLA)

Please cite the original papers and repositories when using the corresponding
methods.
