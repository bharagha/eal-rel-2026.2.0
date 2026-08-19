# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/collect_results.py merging logic (no live docker/GPU required)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = TOOL_ROOT / "scripts" / "collect_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("collect_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["collect_results"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


collect_results = _load_module()


def test_summarize_resources_with_docker_and_gpu_samples() -> None:
    samples = [
        {
            "timestamp": "2026-01-01T00:00:00.000Z",
            "docker_stats": {"CPUPerc": "45.20%", "MemPerc": "10.10%"},
            "gpu": {"engines": {"Render/3D": {"busy": 55.0}, "Video": {"busy": 12.0}}},
        },
        {
            "timestamp": "2026-01-01T00:00:02.000Z",
            "docker_stats": {"CPUPerc": "60.00%", "MemPerc": "12.00%"},
            "gpu": {"engines": {"Render/3D": {"busy": 70.0}, "Video": {"busy": 15.0}}},
        },
    ]

    result = collect_results.summarize_resources(samples)

    assert result["cpu_percent"]["min"] == 45.2
    assert result["cpu_percent"]["max"] == 60.0
    assert result["mem_percent"]["mean"] == 11.05
    assert result["gpu_util_percent"]["max"] == 70.0
    assert result["sample_count"] == 2


def test_summarize_resources_handles_missing_gpu_gracefully() -> None:
    samples = [{"timestamp": "t", "docker_stats": {"CPUPerc": "10%", "MemPerc": "5%"}, "gpu": None}]
    result = collect_results.summarize_resources(samples)
    assert result["gpu_util_percent"] is None
    assert result["cpu_percent"]["mean"] == 10.0


def test_summarize_resources_handles_empty_samples() -> None:
    result = collect_results.summarize_resources([])
    assert result["cpu_percent"] is None
    assert result["mem_percent"] is None
    assert result["gpu_util_percent"] is None
    assert result["sample_count"] == 0


def test_build_summary_extracts_known_performance_keys() -> None:
    benchmark = {
        "mean_ttft_ms": 123.4,
        "mean_tpot_ms": 12.3,
        "request_throughput": 5.5,
        "unrelated_field": "ignored",
    }
    resources = {"cpu_percent": None, "mem_percent": None, "gpu_util_percent": None, "sample_count": 0}

    summary = collect_results.build_summary("ovms", "llm", benchmark, resources)

    assert summary["engine"] == "ovms"
    assert summary["model_type"] == "llm"
    assert summary["performance"]["mean_ttft_ms"] == 123.4
    assert "unrelated_field" not in summary["performance"]


def test_render_table_includes_engine_and_metrics() -> None:
    summary = {
        "engine": "vllm",
        "model_type": "vlm",
        "performance": {"mean_ttft_ms": 100.0, "mean_tpot_ms": 10.0},
        "resources": {
            "cpu_percent": {"min": 1.0, "max": 2.0, "mean": 1.5, "samples": 2},
            "mem_percent": None,
            "gpu_util_percent": None,
        },
    }
    table = collect_results.render_table(summary)
    assert "vllm" in table
    assert "vlm" in table
    assert "mean_ttft_ms" in table


def test_end_to_end_json_output(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark_serving.json"
    benchmark_path.write_text(json.dumps({"mean_ttft_ms": 50.0, "mean_tpot_ms": 5.0, "completed": 10}))

    resources_path = tmp_path / "resources.jsonl"
    resources_path.write_text(
        "\n".join(
            [
                json.dumps({"docker_stats": {"CPUPerc": "20%", "MemPerc": "3%"}, "gpu": None}),
                json.dumps({"docker_stats": {"CPUPerc": "30%", "MemPerc": "4%"}, "gpu": None}),
            ]
        )
    )

    benchmark = collect_results.load_benchmark(benchmark_path)
    resource_samples = collect_results.load_resource_samples(resources_path)
    resources = collect_results.summarize_resources(resource_samples)
    summary = collect_results.build_summary("ovms", "llm", benchmark, resources)

    assert summary["performance"]["completed"] == 10
    assert summary["resources"]["cpu_percent"]["mean"] == 25.0
