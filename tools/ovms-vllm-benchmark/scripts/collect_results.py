#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Render benchmark_serving.py output as a summary report (JSON + table).

Usage:
    python3 collect_results.py \
        --engine ovms --model-type llm \
        --benchmark-json results/ovms_llm_20260101T000000/benchmark_serving.json \
        --config config/models.yaml \
        --output results/ovms_llm_20260101T000000/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_benchmark(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def model_metadata(engine: str, model_type: str, config: dict) -> dict:
    model = config["models"][model_type]
    weight_format = "none"
    if engine == "ovms":
        export_args = model.get("ovms", {}).get("export_extra_args", [])
        for index, arg in enumerate(export_args):
            if arg == "--weight-format" and index + 1 < len(export_args):
                weight_format = str(export_args[index + 1])
                break
            if isinstance(arg, str) and arg.startswith("--weight-format="):
                weight_format = arg.split("=", 1)[1]
                break
    return {"model_name": model["hf_repo"], "weight_format": weight_format}


def build_summary(engine: str, model_type: str, benchmark: dict, metadata: dict) -> dict:
    perf_keys = (
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "request_throughput",
        "output_throughput",
        "completed",
        "duration",
    )
    performance = {k: benchmark[k] for k in perf_keys if k in benchmark}
    return {
        "engine": engine,
        "model_type": model_type,
        **metadata,
        "performance": performance,
    }


def render_table(summary: dict) -> str:
    perf = summary["performance"]
    lines = [
        f"Engine:      {summary['engine']}",
        f"Model type:  {summary['model_type']}",
        f"Model name:  {summary['model_name']}",
        f"Weight format: {summary['weight_format']}",
        "--- Performance ---",
    ]
    for key in ("mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "request_throughput", "output_throughput", "completed"):
        if key in perf:
            lines.append(f"  {key:<22}: {perf[key]}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["ovms", "vllm"])
    parser.add_argument("--model-type", required=True, choices=["llm", "vlm", "moe"])
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_json)
    with args.config.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    metadata = model_metadata(args.engine, args.model_type, config)

    summary = build_summary(args.engine, args.model_type, benchmark, metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    table = render_table(summary)
    print(table)
    (args.output.parent / "summary.txt").write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
