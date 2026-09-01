#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Downloads (and, for OVMS, converts to OpenVINO IR) a preselected model
# ready for serving by either engine.
#
# Usage:
#   scripts/prepare_model.sh <ovms|vllm> <llm|vlm|moe> [precision]
#
# [precision] only applies to the vLLM path (bf16|int4, default bf16) and
# selects which entry under models.<type>.vllm.precisions.* to download from;
# it is ignored for the OVMS path, which is unaffected by this parameter and
# keeps using models.<type>.hf_repo / ovms.export_extra_args as before.
#
# Env vars:
#   HUGGING_FACE_HUB_TOKEN  - required for gated models (e.g. the MoE entry).
#   MODELS_DIR              - override local model cache directory.
#   OVMS_EXPORT_MODEL_REF   - git ref of openvinotoolkit/model_server to pull
#                             export_model.py from (pin to match the served
#                             OVMS image). Defaults to a tag matching 2026.3.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> <llm|vlm|moe> [precision]}
MODEL_TYPE=${2:?Usage: $0 <ovms|vllm> <llm|vlm|moe> [precision]}
PRECISION=${3:-bf16}
OVMS_EXPORT_MODEL_REF=${OVMS_EXPORT_MODEL_REF:-releases/2026/3}

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "engine must be 'ovms' or 'vllm', got: ${ENGINE}"

SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
GATED=$(yaml_get "models.${MODEL_TYPE}.gated" "false")

if [[ "${ENGINE}" == "vllm" ]]; then
  require_cmd huggingface-cli
  HF_REPO=$(yaml_get "models.${MODEL_TYPE}.vllm.precisions.${PRECISION}.hf_repo")
  if [[ "${GATED}" == "true" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    die "model ${HF_REPO} is gated; set HUGGING_FACE_HUB_TOKEN before running"
  fi
  mkdir -p "${MODELS_DIR}"
  log "Downloading ${HF_REPO} (precision: ${PRECISION}) into HF cache for vLLM serving (served as '${SERVED_NAME}')"
  huggingface-cli download "${HF_REPO}" \
    --cache-dir "${MODELS_DIR}/hf-cache" \
    ${HUGGING_FACE_HUB_TOKEN:+--token "${HUGGING_FACE_HUB_TOKEN}"}
  log "Download complete: ${MODELS_DIR}/hf-cache"
  exit 0
fi

# --- OVMS path (unaffected by --precision) ---
HF_REPO=$(yaml_get "models.${MODEL_TYPE}.hf_repo")

if [[ "${GATED}" == "true" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  die "model ${HF_REPO} is gated; set HUGGING_FACE_HUB_TOKEN before running"
fi

mkdir -p "${MODELS_DIR}"

# --- OVMS path: download export_model.py (pinned) and convert to IR ---
require_cmd python3
require_cmd pip3

EXPORT_SCRIPT_DIR="${TOOL_ROOT}/.export-tool"
EXPORT_SCRIPT="${EXPORT_SCRIPT_DIR}/export_model.py"
mkdir -p "${EXPORT_SCRIPT_DIR}"

if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  log "Fetching export_model.py (ref: ${OVMS_EXPORT_MODEL_REF}) from openvinotoolkit/model_server"
  curl -fsSL \
    "https://raw.githubusercontent.com/openvinotoolkit/model_server/${OVMS_EXPORT_MODEL_REF}/demos/common/export_models/export_model.py" \
    -o "${EXPORT_SCRIPT}"
  curl -fsSL \
    "https://raw.githubusercontent.com/openvinotoolkit/model_server/${OVMS_EXPORT_MODEL_REF}/demos/common/export_models/requirements.txt" \
    -o "${EXPORT_SCRIPT_DIR}/requirements.txt" || true
fi

if [[ -f "${EXPORT_SCRIPT_DIR}/requirements.txt" ]]; then
  pip3 install --quiet -r "${EXPORT_SCRIPT_DIR}/requirements.txt"
fi

OUT_DIR="${MODELS_DIR}/${SERVED_NAME}"
mkdir -p "${OUT_DIR}"

# Read extra per-model export args (e.g. weight-format) as a JSON array and
# expand into individual argv entries.
EXTRA_ARGS_JSON=$(yaml_get "models.${MODEL_TYPE}.ovms.export_extra_args" "[]")
mapfile -t EXTRA_ARGS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${EXTRA_ARGS_JSON}")

log "Exporting ${HF_REPO} to OpenVINO IR at ${OUT_DIR} (extra args: ${EXTRA_ARGS[*]:-none})"
python3 "${EXPORT_SCRIPT}" text_generation \
  --source_model "${HF_REPO}" \
  --model_repository_path "${MODELS_DIR}" \
  --model_name "${SERVED_NAME}" \
  --target_device GPU \
  ${HUGGING_FACE_HUB_TOKEN:+--hf_token "${HUGGING_FACE_HUB_TOKEN}"} \
  "${EXTRA_ARGS[@]}"

log "OVMS model export complete: ${OUT_DIR}"
