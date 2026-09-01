# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for config/models.yaml parsing (via scripts/yaml_get.py)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TOOL_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = TOOL_ROOT / "config" / "models.yaml"
YAML_GET_SCRIPT = TOOL_ROOT / "scripts" / "yaml_get.py"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("model_type", ["llm", "vlm", "moe", "minicpm"])
def test_model_entry_has_required_fields(config: dict, model_type: str) -> None:
    model_cfg = config["models"][model_type]
    assert model_cfg["hf_repo"]
    assert model_cfg["model_type"] in ("llm", "vlm", "moe")
    assert model_cfg["served_model_name"]
    assert "dataset" in model_cfg
    assert model_cfg["dataset"]["hf_dataset"]
    assert "gated" in model_cfg


def test_expected_model_ids(config: dict) -> None:
    assert config["models"]["llm"]["hf_repo"] == "microsoft/Phi-4-mini-instruct"
    assert config["models"]["vlm"]["hf_repo"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert config["models"]["moe"]["hf_repo"] == "google/gemma-4-26B-A4B-it"
    assert config["models"]["minicpm"]["hf_repo"] == "openbmb/MiniCPM-V-4_5"


def test_minicpm_model_entry(config: dict) -> None:
    minicpm_cfg = config["models"]["minicpm"]
    assert minicpm_cfg["served_model_name"] == "minicpm-v-4_5"
    assert minicpm_cfg["model_type"] == "vlm"
    assert minicpm_cfg["gated"] is False
    assert minicpm_cfg["dataset"]["name"] == "vqa"


def test_moe_model_is_gated(config: dict) -> None:
    assert config["models"]["moe"]["gated"] is True


@pytest.mark.parametrize("engine", ["ovms", "vllm"])
def test_engine_entry_has_required_fields(config: dict, engine: str) -> None:
    engine_cfg = config["engines"][engine]
    assert engine_cfg["image"]
    assert engine_cfg["port"]
    assert engine_cfg["health_path"]


@pytest.mark.parametrize("model_type", ["llm", "vlm", "moe", "minicpm"])
def test_vllm_precisions_bf16_required(config: dict, model_type: str) -> None:
    """Every model must define at least a bf16 vLLM precision entry."""
    vllm_cfg = config["models"][model_type]["vllm"]
    assert vllm_cfg["default_precision"] == "bf16"
    bf16_cfg = vllm_cfg["precisions"]["bf16"]
    assert bf16_cfg["hf_repo"]
    assert isinstance(bf16_cfg["server_extra_args"], list)


@pytest.mark.parametrize("model_type", ["llm", "vlm", "moe", "minicpm"])
def test_vllm_int4_precision_uses_torchao_for_all_models(config: dict, model_type: str) -> None:
    """int4 for vLLM is produced on-the-fly via torchao for every model
    (no more separate pre-quantized HF repo / placeholder)."""
    int4_cfg = config["models"][model_type]["vllm"]["precisions"]["int4"]
    assert int4_cfg["quantization"] == "torchao"
    assert "hf_repo" not in int4_cfg
    assert isinstance(int4_cfg["server_extra_args"], list)
    assert "--quantization" in int4_cfg["server_extra_args"]
    assert "torchao" in int4_cfg["server_extra_args"]


@pytest.mark.parametrize("model_type", ["llm", "vlm", "moe", "minicpm"])
def test_ovms_precisions_bf16_and_int4_defined_for_all_models(config: dict, model_type: str) -> None:
    """OVMS bf16 (mapped to fp16 weight-format) and int4 must be defined for
    every model, mirroring the vLLM precisions structure."""
    ovms_precisions = config["models"][model_type]["ovms"]["precisions"]
    assert ovms_precisions["bf16"]["export_extra_args"] == ["--weight-format", "fp16"]
    assert ovms_precisions["int4"]["export_extra_args"] == ["--weight-format", "int4"]


def test_yaml_get_script_returns_scalar() -> None:
    result = subprocess.run(
        [sys.executable, str(YAML_GET_SCRIPT), str(CONFIG_PATH), "models.llm.hf_repo"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "microsoft/Phi-4-mini-instruct"


def test_yaml_get_script_returns_json_for_list() -> None:
    import json

    result = subprocess.run(
        [sys.executable, str(YAML_GET_SCRIPT), str(CONFIG_PATH), "models.llm.ovms.precisions.bf16.export_extra_args"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["--weight-format", "fp16"]


def test_yaml_get_script_missing_key_uses_default() -> None:
    result = subprocess.run(
        [sys.executable, str(YAML_GET_SCRIPT), str(CONFIG_PATH), "models.llm.does_not_exist", "--default", "fallback"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "fallback"


def test_yaml_get_script_missing_key_no_default_errors() -> None:
    result = subprocess.run(
        [sys.executable, str(YAML_GET_SCRIPT), str(CONFIG_PATH), "models.llm.does_not_exist"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "error" in result.stderr
