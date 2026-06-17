#!/usr/bin/env python3
"""Unified launcher for original / VCD / AVISC / AGLA experiments.

Run this file from anywhere inside the VCD project. It selects the existing
experiment implementation from:

- method: original, vcd, avisc, agla
- model: llava, instructblip2/blip2, internvl
- benchmark: chair, pope, mme
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_ROOT = Path(__file__).resolve().parent
RUNNERS_ROOT = UNIFIED_ROOT / "runners"

MODELS = {
    "llava": "llava",
    "instructblip2": "blip2",
    "blip2": "blip2",
    "internvl": "internvl",
}

POPE_DATASETS = ("coco", "aokvqa", "gqa")
POPE_QUESTIONS = ("random", "popular", "adversarial")
MME_NAMES = ("existence", "count", "position", "OCR", "color")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def find_pope_root() -> Path | None:
    candidates = [
        UNIFIED_ROOT / "pope_dataset",
        UNIFIED_ROOT / "outputs" / "pope_dataset",
        PROJECT_ROOT / "pope_dataset",
        Path("/data/dtt/projects/SPAC/pope_dataset"),
        Path("/data/dtt/projects/VAME/pope_dataset"),
    ]
    for candidate in candidates:
        if (candidate / "POPE").is_dir():
            return candidate
    return None


def method_args(method: str) -> list[str]:
    if method == "original":
        return ["--original"]
    if method == "avisc":
        return ["--use_avisc", "True"]
    return []


def script_for(benchmark: str, model: str, method: str) -> Path:
    if method == "agla":
        if benchmark == "mme" and model == "internvl":
            raise SystemExit("MME + internvl is not wired for AGLA in the original AGLA scripts.")
        return RUNNERS_ROOT / "agla_runner.py"
    if benchmark == "pope":
        return RUNNERS_ROOT / "pope_vcd_total.py"
    if benchmark == "mme":
        if model == "internvl":
            raise SystemExit("MME + internvl is not wired in the current src scripts.")
        if method == "vcd":
            return RUNNERS_ROOT / "vcd" / "vcd_total_mme.py"
        return RUNNERS_ROOT / "avisc" / "avisc_total_mme.py"
    if benchmark == "chair":
        if model == "internvl":
            return RUNNERS_ROOT / "internvl_avisc_vcd_chair.py"
        if model == "llava":
            return RUNNERS_ROOT / "llava_avisc_vcd_chair.py"
        return RUNNERS_ROOT / "avisc" / "blip2_avisc_vcd_chair.py"
    raise SystemExit(f"unknown benchmark: {benchmark}")


def build_base_command(args: argparse.Namespace, script: Path) -> list[str]:
    command = [sys.executable, str(script)]
    command.extend(method_args(args.method))

    if args.method == "agla":
        command.extend(["--benchmark", args.benchmark])
        command.extend(["--model_name", args.model])
    elif args.benchmark in {"pope", "mme"}:
        command.extend(["--model_name", args.model])

    if args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.max_new_tokens is not None:
        command.extend(["--max_new_tokens", str(args.max_new_tokens)])
    if args.max_length is not None:
        command.extend(["--max_length", str(args.max_length)])
    if args.num_beams is not None:
        command.extend(["--num_beams", str(args.num_beams)])
    if args.do_sample:
        command.append("--do_sample")

    command.extend(["--cd_alpha", str(args.cd_alpha), "--cd_beta", str(args.cd_beta)])
    if args.method != "agla":
        command.extend(["--layer_gamma", str(args.layer_gamma)])
        command.extend(["--masking_scheme", args.masking_scheme])
        if args.lamb is not None:
            command.extend(["--lamb", str(args.lamb)])

    if args.image_folder:
        command.extend(["--image_folder", str(args.image_folder)])
    if args.model == "internvl" and args.internvl_model_path:
        command.extend(["--internvl_model_path", str(args.internvl_model_path)])
    if args.specific_name:
        command.extend(["--specific_name", args.specific_name])

    command.extend(args.extra_args)
    return command


def expand_commands(args: argparse.Namespace) -> list[list[str]]:
    script = script_for(args.benchmark, args.model, args.method)
    base = build_base_command(args, script)

    if args.benchmark == "pope":
        datasets = POPE_DATASETS if args.all else (args.type_dataset,)
        questions = POPE_QUESTIONS if args.all else (args.type_question,)
        return [
            base + ["--type_dataset", dataset, "--type_question", question]
            for dataset in datasets
            for question in questions
        ]

    if args.benchmark == "mme":
        names = MME_NAMES if args.all else (args.mme_name,)
        return [base + ["--mme_name", name] for name in names]

    return [base]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments by selecting method, model, and benchmark."
    )
    parser.add_argument("--method", choices=["original", "vcd", "avisc", "agla"], required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--benchmark", choices=["chair", "pope", "mme"], required=True)

    parser.add_argument("--all", action="store_true", help="Run all POPE/MME subsets.")
    parser.add_argument("--type_dataset", choices=POPE_DATASETS, default="coco")
    parser.add_argument("--type_question", choices=POPE_QUESTIONS, default="popular")
    parser.add_argument("--mme_name", choices=MME_NAMES, default="existence")

    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Only run the first N samples in each selected benchmark split.",
    )

    parser.add_argument("--image_folder", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", default="zeros")
    parser.add_argument("--lamb", type=int, default=None)
    parser.add_argument(
        "--internvl_model_path",
        type=Path,
        default=Path("/data/dtt/pretrain_model_or_weight/InternVL2-2B"),
    )
    parser.add_argument("--specific_name", default=None)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to the underlying experiment script.",
    )
    args = parser.parse_args()
    args.model = MODELS[args.model]
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    return args


def main() -> int:
    args = parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.limit_samples is not None:
        env["VCD_SAMPLE_LIMIT"] = str(args.limit_samples)
    env.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    env["PYTHONPATH"] = str(UNIFIED_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    pope_root = find_pope_root()
    if pope_root:
        env.setdefault("POPE_ROOT", str(pope_root))

    commands = expand_commands(args)
    for command in commands:
        if args.limit_samples is not None:
            print(f"# VCD_SAMPLE_LIMIT={args.limit_samples}", flush=True)
        if env.get("HF_ENDPOINT"):
            print(f"# HF_ENDPOINT={env['HF_ENDPOINT']}", flush=True)
        print("+ " + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=str(UNIFIED_ROOT), env=env)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
