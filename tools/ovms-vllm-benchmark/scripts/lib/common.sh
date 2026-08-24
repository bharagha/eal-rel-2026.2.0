#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Shared shell helpers for the OVMS vs vLLM benchmark scripts.
# Source this file from other scripts: `source "$(dirname "$0")/lib/common.sh"`

set -euo pipefail

# Resolve absolute path to the tool's root directory (parent of scripts/).
# This file lives in scripts/lib/, so go up two levels to reach the tool root.
TOOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${TOOL_ROOT}/config/models.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${TOOL_ROOT}/results}"
MODELS_DIR="${MODELS_DIR:-${TOOL_ROOT}/model-cache}"

# Activate the tool's virtualenv so python3/pip3/huggingface-cli resolve to the
# pinned benchmark dependencies rather than the system ones. The venv is
# bootstrapped (created + requirements installed) on first use if missing, so
# it is safe for the benchmark cleanup step to delete it between runs.
ensure_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    return 0
  fi
  local venv_dir="${TOOL_ROOT}/venv"
  if [[ ! -f "${venv_dir}/bin/activate" ]]; then
    command -v python3 >/dev/null 2>&1 || die "required command not found: python3"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating virtualenv at ${venv_dir}" >&2
    python3 -m venv "${venv_dir}" || die "failed to create virtualenv at ${venv_dir}"
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
    python3 -m pip install --quiet --upgrade pip
    if [[ -f "${TOOL_ROOT}/requirements.txt" ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing requirements into virtualenv" >&2
      python3 -m pip install --quiet -r "${TOOL_ROOT}/requirements.txt" \
        || die "failed to install requirements into ${venv_dir}"
    fi
  else
    # shellcheck disable=SC1091
    source "${venv_dir}/bin/activate"
  fi
}


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
  if [[ $# -ge 2 ]]; then
    python3 "${TOOL_ROOT}/scripts/yaml_get.py" "${CONFIG_FILE}" "${dotted_path}" --default "$2"
  else
    python3 "${TOOL_ROOT}/scripts/yaml_get.py" "${CONFIG_FILE}" "${dotted_path}"
  fi
}

require_cmd() {
  local cmd=$1
  command -v "${cmd}" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

# container_name <engine>
container_name() {
  echo "ovms-vllm-bench-$1"
}

# Bootstrap/activate the virtualenv now that log/die helpers are defined.
ensure_venv
