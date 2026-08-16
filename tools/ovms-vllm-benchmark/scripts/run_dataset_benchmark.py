#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dataset-aware wrapper around vLLM's official benchmark_serving.py.

Selects the right dataset for the model type (ShareGPT-style text prompts
for `llm`/`moe`, a VQA-style image+prompt dataset for `vlm`) and invokes
vLLM's benchmark_serving.py against an OpenAI-compatible endpoint (served by
either OVMS or vLLM), capturing TTFT/TPOT/throughput metrics.

vLLM's benchmark_serving.py is fetched (pinned to a version compatible with
the intel/vllm:0.21.0-xpu image) on first use and cached locally, avoiding a
hard dependency on having the full vllm source checked out.

Usage:
    python3 run_dataset_benchmark.py \
        --base-url http://localhost:8000 \
        --model-type llm \
        --served-model-name phi-4-mini-instruct \
        --config ../config/models.yaml \
        --output ../results/ovms_llm_20260101T000000/benchmark_serving.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

TOOL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_SCRIPT_REF = "v0.21.0"
BENCH_SCRIPT_URL_TMPL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/{ref}/benchmarks/benchmark_serving.py"
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_benchmark_script(cache_dir: Path, ref: str) -> Path:
    """Download (once) vLLM's benchmark_serving.py, pinned to `ref`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    script_path = cache_dir / f"benchmark_serving_{ref}.py"
    if script_path.exists():
        return script_path

    url = BENCH_SCRIPT_URL_TMPL.format(ref=ref)
    print(f"Fetching vLLM benchmark_serving.py ({ref}) -> {script_path}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted, pinned GitHub raw URL)
        script_path.write_bytes(resp.read())
    return script_path


def dataset_args_for(model_type: str, dataset_cfg: dict) -> list[str]:
    """Build benchmark_serving.py --dataset-name/--dataset-path style args."""
    hf_dataset = dataset_cfg["hf_dataset"]
    if model_type == "vlm":
        # benchmark_serving.py's "hf" dataset backend supports multimodal
        # (image+prompt) HF datasets such as VQA-style collections.
        return ["--dataset-name", "hf", "--dataset-path", hf_dataset, "--hf-split", dataset_cfg.get("split", "test")]
    # llm / moe: ShareGPT-style text-only conversations.
    return ["--dataset-name", "hf", "--dataset-path", hf_dataset, "--hf-split", dataset_cfg.get("split", "train")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--model-type", required=True, choices=["llm", "vlm", "moe"])
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--config", type=Path, default=TOOL_ROOT / "config" / "models.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Path to write benchmark_serving.py JSON result")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--bench-script-ref", default=DEFAULT_BENCH_SCRIPT_REF)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw arg to pass through to benchmark_serving.py (repeatable)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["models"][args.model_type]
    dataset_cfg = model_cfg["dataset"]

    cache_dir = TOOL_ROOT / ".bench-tool"
    bench_script = ensure_benchmark_script(cache_dir, args.bench_script_ref)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(bench_script),
        "--backend",
        "openai-chat",
        "--base-url",
        args.base_url,
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        args.served_model_name,
        "--num-prompts",
        str(args.num_prompts),
        "--save-result",
        "--result-filename",
        str(args.output),
        *dataset_args_for(args.model_type, dataset_cfg),
        *args.extra_arg,
    ]

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=str(bench_script.parent), check=False)
    if result.returncode != 0:
        print(f"benchmark_serving.py exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    if not args.output.exists():
        print(f"Expected result file not found: {args.output}", file=sys.stderr)
        return 1

    with args.output.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    print(json.dumps({k: summary.get(k) for k in ("mean_ttft_ms", "mean_tpot_ms", "request_throughput") if k in summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
