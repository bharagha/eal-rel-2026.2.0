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
# Precision suffix is only meaningful for vLLM (OVMS precision is governed
# solely by the existing ovms.export_extra_args, not by --precision), but the
# suffix is applied whenever a non-empty precision is passed so that bf16
# and int4 vLLM runs don't collide on the same container name.
container_name() {
  local engine=$1
  local precision=${2:-}
  if [[ -n "${precision}" && "${engine}" == "vllm" ]]; then
    echo "ovms-vllm-bench-${engine}-${precision}"
  else
    echo "ovms-vllm-bench-${engine}"
  fi
}
