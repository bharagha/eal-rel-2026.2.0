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


def test_build_summary_extracts_known_performance_keys() -> None:
    benchmark = {
        "mean_ttft_ms": 123.4,
        "mean_tpot_ms": 12.3,
        "request_throughput": 5.5,
        "unrelated_field": "ignored",
    }
    metadata = {"model_name": "microsoft/Phi-4-mini-instruct", "weight_format": "int4"}

    summary = collect_results.build_summary("ovms", "llm", benchmark, metadata)

    assert summary["engine"] == "ovms"
    assert summary["model_type"] == "llm"
    assert summary["model_name"] == "microsoft/Phi-4-mini-instruct"
    assert summary["weight_format"] == "int4"
    assert summary["performance"]["mean_ttft_ms"] == 123.4
    assert "unrelated_field" not in summary["performance"]


def test_render_table_includes_engine_and_metrics() -> None:
    summary = {
        "engine": "vllm",
        "model_type": "vlm",
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "weight_format": "none",
        "performance": {"mean_ttft_ms": 100.0, "mean_tpot_ms": 10.0},
    }
    table = collect_results.render_table(summary)
    assert "vllm" in table
    assert "vlm" in table
    assert "mean_ttft_ms" in table
    assert "Qwen/Qwen3-VL-8B-Instruct" in table
    assert "none" in table


def test_model_metadata_uses_ovms_weight_format_and_vllm_none() -> None:
    config = {
        "models": {
            "llm": {
                "hf_repo": "microsoft/Phi-4-mini-instruct",
                "ovms": {"export_extra_args": ["--weight-format", "int4"]},
            }
        }
    }

    assert collect_results.model_metadata("ovms", "llm", config)["weight_format"] == "int4"
    assert collect_results.model_metadata("vllm", "llm", config)["weight_format"] == "none"
