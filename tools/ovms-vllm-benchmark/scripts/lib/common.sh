#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Shared shell helpers for the OVMS vs vLLM benchmark scripts.
# Source this file from other scripts: `source "$(dirname "$0")/lib/common.sh"`

set -euo pipefail

# Resolve absolute path to the tool's root directory (parent of scripts/).
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${TOOL_ROOT}/config/models.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${TOOL_ROOT}/results}"
MODELS_DIR="${MODELS_DIR:-${TOOL_ROOT}/model-cache}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

# yaml_get <dotted.path> [default]
yaml_get() {
  local dotted_path=$1
  local default=${2:-}
  if [[ -n "${default}" ]]; then
    python3 "${TOOL_ROOT}/scripts/yaml_get.py" "${CONFIG_FILE}" "${dotted_path}" --default "${default}"
  else
    python3 "${TOOL_ROOT}/scripts/yaml_get.py" "${CONFIG_FILE}" "${dotted_path}"
  fi
}

require_cmd() {
  local cmd=$1
  command -v "${cmd}" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

# container_name <engine> [precision]
#
# Both engines now support bf16/int4 precisions (OVMS via per-precision IR
# export directories, vLLM via HF repo vs. torchao-converted checkpoint), so
# the precision suffix is applied for either engine whenever a non-empty
# precision is passed, keeping bf16 and int4 runs from colliding on the same
# container name.
container_name() {
  local engine=$1
  local precision=${2:-}
  if [[ -n "${precision}" ]]; then
    echo "ovms-vllm-bench-${engine}-${precision}"
  else
    echo "ovms-vllm-bench-${engine}"
  fi
}
