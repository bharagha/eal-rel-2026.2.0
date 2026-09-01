#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Quantize a bf16 Hugging Face checkpoint to int4 (weight-only) using torchao.

Intended to be run *inside* the same container image used to serve vLLM
(`intel/vllm:...-xpu`), which already ships a compatible PyTorch build, via:

    docker run --rm \
      -v "$MODELS_DIR/hf-cache:/root/.cache/huggingface" \
      -v "$MODELS_DIR/torchao-int4:/out" \
      -e HUGGING_FACE_HUB_TOKEN \
      "$VLLM_IMAGE" \
      python3 /path/to/convert_torchao.py --hf-repo <repo> --model-type <type> \
        --output-dir /out/<served_model_name>

The resulting directory is a standard Hugging Face checkpoint (config +
weights + tokenizer/processor) with a torchao int4 weight-only quantization
applied, suitable for `vllm serve <output-dir> --quantization torchao`.

This script intentionally avoids adding torch/transformers/torchao to the
benchmark tool's own (host-side) requirements.txt: it only ever executes
inside the vLLM serving container, which already depends on torch and
transformers, and installs torchao on demand if missing.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# Model types that require trust_remote_code and a vision-language model
# loading path (custom modeling code shipped with the repo).
VLM_MODEL_TYPES = {"vlm", "minicpm"}


def _ensure_torchao() -> None:
    try:
        importlib.import_module("torchao")
    except ImportError:
        print("[convert_torchao] torchao not found, installing...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "torchao"],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo", required=True, help="Source bf16 Hugging Face repo id")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["llm", "vlm", "moe", "minicpm"],
        help="Logical model type (determines the transformers loading path)",
    )
    parser.add_argument("--output-dir", required=True, help="Local directory to write the quantized checkpoint to")
    parser.add_argument("--hf-token", default=None, help="Hugging Face access token for gated repos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    marker = output_dir / ".torchao-int4-complete"
    if marker.exists():
        print(f"[convert_torchao] {output_dir} already converted, skipping.", file=sys.stderr)
        return 0

    _ensure_torchao()

    import torch
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from transformers import AutoProcessor, AutoTokenizer

    token = args.hf_token or None
    common_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.model_type in VLM_MODEL_TYPES,
        token=token,
    )

    print(f"[convert_torchao] loading {args.hf_repo} (model_type={args.model_type})", file=sys.stderr)
    if args.model_type in VLM_MODEL_TYPES:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(args.hf_repo, **common_kwargs)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.hf_repo, **common_kwargs)

    print("[convert_torchao] applying torchao Int4WeightOnlyConfig quantization", file=sys.stderr)
    quantize_(model, Int4WeightOnlyConfig())

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[convert_torchao] saving quantized checkpoint to {output_dir}", file=sys.stderr)
    model.save_pretrained(output_dir, safe_serialization=False)

    # Persist tokenizer/processor alongside the quantized weights so the
    # output directory is a self-contained checkpoint vLLM can load directly.
    try:
        processor = AutoProcessor.from_pretrained(args.hf_repo, trust_remote_code=args.model_type in VLM_MODEL_TYPES, token=token)
        processor.save_pretrained(output_dir)
    except Exception as exc:  # noqa: BLE001 - processor is optional for text-only models
        print(f"[convert_torchao] no processor to save ({exc}); saving tokenizer only", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, trust_remote_code=args.model_type in VLM_MODEL_TYPES, token=token)
        tokenizer.save_pretrained(output_dir)

    marker.write_text("ok\n", encoding="utf-8")
    print("[convert_torchao] done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
