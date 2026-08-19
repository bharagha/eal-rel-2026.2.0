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
# vLLM moved benchmark_serving.py into the `vllm bench serve` CLI in newer
# releases (the file at recent tags is just a deprecation stub). Pin to the
# last self-contained standalone version, which only needs a sibling
# backend_request_func.py and no importable `vllm` package.
DEFAULT_BENCH_SCRIPT_REF = "v0.6.6"
BENCH_SCRIPT_URL_TMPL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/{ref}/benchmarks/benchmark_serving.py"
)
# Sibling modules that benchmark_serving.py imports from its own directory and
# that must be fetched alongside it.
BENCH_SCRIPT_SIBLINGS = ("backend_request_func.py",)
BENCH_SIBLING_URL_TMPL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/{ref}/benchmarks/{name}"
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _patch_sibling(sibling_path: Path) -> None:
    """Make the standalone benchmark helpers tolerant of vLLM-only kwargs.

    benchmark_serving.py is written assuming vLLM's own get_tokenizer (which
    accepts e.g. ``tokenizer_mode``) is importable, and only falls back to the
    standalone ``backend_request_func.get_tokenizer`` when ``vllm`` is not
    installed. That fallback has a narrower signature, so calls made with
    ``tokenizer_mode=...`` raise ``TypeError``. Relax the signature to accept
    and ignore any extra keyword arguments. Idempotent.
    """
    if sibling_path.name != "backend_request_func.py":
        return
    text = sibling_path.read_text(encoding="utf-8")
    needle = "    pretrained_model_name_or_path: str, trust_remote_code: bool\n"
    replacement = (
        "    pretrained_model_name_or_path: str,\n"
        "    trust_remote_code: bool = False,\n"
        "    **kwargs,\n"
    )
    if needle in text:
        sibling_path.write_text(text.replace(needle, replacement), encoding="utf-8")


def ensure_benchmark_script(cache_dir: Path, ref: str) -> Path:
    """Download (once) vLLM's benchmark_serving.py + siblings, pinned to `ref`."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # benchmark_serving.py imports its siblings by bare module name, so they
    # must sit next to it under a ref-specific name-independent filename.
    for name in BENCH_SCRIPT_SIBLINGS:
        sibling_path = cache_dir / name
        if not sibling_path.exists():
            url = BENCH_SIBLING_URL_TMPL.format(ref=ref, name=name)
            print(f"Fetching vLLM {name} ({ref}) -> {sibling_path}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted, pinned GitHub raw URL)
                sibling_path.write_bytes(resp.read())
            _patch_sibling(sibling_path)

    script_path = cache_dir / f"benchmark_serving_{ref}.py"
    if script_path.exists():
        return script_path

    url = BENCH_SCRIPT_URL_TMPL.format(ref=ref)
    print(f"Fetching vLLM benchmark_serving.py ({ref}) -> {script_path}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 (trusted, pinned GitHub raw URL)
        script_path.write_bytes(resp.read())
    return script_path


def ensure_sharegpt_dataset(hf_dataset: str, cache_dir: Path, filename: str) -> Path:
    """Download (once) the ShareGPT conversations JSON from a HF dataset repo.

    benchmark_serving.py's ``--dataset-name sharegpt`` path reads a local JSON
    file of ShareGPT-style conversations, rather than a `datasets`-loadable
    repo. ``anon8231489123/ShareGPT_Vicuna_unfiltered`` ships this as a single
    large JSON blob that `datasets` cannot auto-infer, so fetch the file
    directly and cache it locally.
    """
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=hf_dataset,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(local)


def dataset_args_for(model_type: str, dataset_cfg: dict, cache_dir: Path) -> list[str]:
    """Build benchmark_serving.py --dataset-name/--dataset-path style args."""
    hf_dataset = dataset_cfg["hf_dataset"]
    if model_type == "vlm":
        # benchmark_serving.py's "hf" dataset backend supports multimodal
        # (image+prompt) HF datasets. MMMU/MMMU_Pro (subset "vision") is
        # handled by a dedicated sampler that base64-embeds each image.
        args = ["--dataset-name", "hf", "--dataset-path", hf_dataset,
                "--hf-split", dataset_cfg.get("split", "test")]
        subset = dataset_cfg.get("subset")
        if subset:
            args += ["--hf-subset", subset]
        return args
    # llm / moe: ShareGPT-style text-only conversations. Use the standalone
    # "sharegpt" sampler with the repo's conversations JSON fetched locally.
    sharegpt_file = dataset_cfg.get("hf_file", "ShareGPT_V3_unfiltered_cleaned_split.json")
    local_path = ensure_sharegpt_dataset(hf_dataset, cache_dir / "datasets", sharegpt_file)
    return ["--dataset-name", "sharegpt", "--dataset-path", str(local_path)]


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
    # The served model name (e.g. "phi-4-mini-instruct") is what the endpoint
    # expects as the request "model", but it is not a resolvable Hugging Face
    # repo. benchmark_serving.py needs a real tokenizer to count prompt/output
    # tokens, so point --tokenizer at the source HF repo.
    hf_repo = model_cfg["hf_repo"]

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
        "--tokenizer",
        hf_repo,
        "--trust-remote-code",
        "--num-prompts",
        str(args.num_prompts),
        "--save-result",
        "--result-filename",
        str(args.output),
        *dataset_args_for(args.model_type, dataset_cfg, cache_dir),
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
