#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Merge benchmark_serving.py output with resource-monitor samples into a
single summary report (JSON + human-readable table) for one benchmark run.

Usage:
    python3 collect_results.py \
        --engine ovms --model-type llm \
        --benchmark-json results/ovms_llm_20260101T000000/benchmark_serving.json \
        --resources-jsonl results/ovms_llm_20260101T000000/resources.jsonl \
        --output results/ovms_llm_20260101T000000/summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_benchmark(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_resource_samples(path: Path) -> list[dict]:
    samples = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def _parse_percent(value: Any) -> float | None:
    """docker stats reports e.g. '12.34%' or '1.2GiB / 3.5GiB'; extract first float."""
    if value is None:
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return None


def summarize_resources(samples: list[dict]) -> dict:
    cpu_values: list[float] = []
    mem_values: list[float] = []
    gpu_util_values: list[float] = []

    for sample in samples:
        stats = sample.get("docker_stats") or {}
        cpu = _parse_percent(stats.get("CPUPerc"))
        mem = _parse_percent(stats.get("MemPerc"))
        if cpu is not None:
            cpu_values.append(cpu)
        if mem is not None:
            mem_values.append(mem)

        gpu = sample.get("gpu")
        if isinstance(gpu, dict):
            # intel_gpu_top JSON has an "engines" section with busy percentages;
            # xpu-smi JSON layout differs. Try a few common shapes defensively.
            util = None
            if "engines" in gpu and isinstance(gpu["engines"], dict):
                busy_vals = [
                    v.get("busy") for v in gpu["engines"].values() if isinstance(v, dict) and "busy" in v
                ]
                busy_vals = [b for b in busy_vals if isinstance(b, (int, float))]
                if busy_vals:
                    util = max(busy_vals)
            if util is not None:
                gpu_util_values.append(float(util))

    def _stats(values: list[float]) -> dict | None:
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "samples": len(values),
        }

    return {
        "cpu_percent": _stats(cpu_values),
        "mem_percent": _stats(mem_values),
        "gpu_util_percent": _stats(gpu_util_values),
        "sample_count": len(samples),
    }


def build_summary(engine: str, model_type: str, benchmark: dict, resources: dict, precision: str | None = None) -> dict:
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
    summary = {
        "engine": engine,
        "model_type": model_type,
        "performance": performance,
        "resources": resources,
    }
    # precision is informational only (vLLM-specific selector); omitted when
    # not provided (e.g. OVMS runs, where it doesn't apply).
    if precision:
        summary["precision"] = precision
    return summary


def render_table(summary: dict) -> str:
    perf = summary["performance"]
    res = summary["resources"]
    lines = [
        f"Engine:      {summary['engine']}",
        f"Model type:  {summary['model_type']}",
    ]
    if summary.get("precision"):
        lines.append(f"Precision:   {summary['precision']}")
    lines.append("--- Performance ---")
    for key in ("mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "request_throughput", "output_throughput", "completed"):
        if key in perf:
            lines.append(f"  {key:<22}: {perf[key]}")
    lines.append("--- Resources ---")
    for key in ("cpu_percent", "mem_percent", "gpu_util_percent"):
        stat = res.get(key)
        if stat:
            lines.append(
                f"  {key:<22}: min={stat['min']:.1f} mean={stat['mean']:.1f} max={stat['max']:.1f} (n={stat['samples']})"
            )
        else:
            lines.append(f"  {key:<22}: n/a")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["ovms", "vllm"])
    parser.add_argument("--model-type", required=True, choices=["llm", "vlm", "moe"])
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--resources-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--precision",
        default=None,
        help="vLLM serving precision used for this run (e.g. bf16, int4); informational, ignored for OVMS",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_json)
    resource_samples = load_resource_samples(args.resources_jsonl)
    resources = summarize_resources(resource_samples)

    summary = build_summary(args.engine, args.model_type, benchmark, resources, args.precision)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    table = render_table(summary)
    print(table)
    (args.output.parent / "summary.txt").write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
