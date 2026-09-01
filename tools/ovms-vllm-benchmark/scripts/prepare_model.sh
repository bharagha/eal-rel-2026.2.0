#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Downloads (and converts, for both engines) a preselected model ready for
# serving by either engine.
#
# Usage:
#   scripts/prepare_model.sh <ovms|vllm> <llm|vlm|moe|minicpm> [precision]
#
# [precision] applies to BOTH engines now (bf16|int4, default bf16):
#   - vllm: bf16 downloads models.<type>.vllm.precisions.bf16.hf_repo as-is;
#           int4 downloads the bf16 repo, then quantizes it to int4 with
#           torchao (scripts/convert_torchao.py, run inside the vLLM serving
#           image) and caches the result under
#           ${MODELS_DIR}/torchao-int4/<served_model_name>.
#   - ovms:  selects models.<type>.ovms.precisions.<precision>.export_extra_args
#            (weight-format) and exports IR into a precision-scoped directory
#            (<served_model_name>-<precision>) so bf16/int4 exports coexist.
#
# Env vars:
#   HUGGING_FACE_HUB_TOKEN  - required for gated models (e.g. the MoE entry).
#   MODELS_DIR              - override local model cache directory.
#   OVMS_EXPORT_MODEL_REF   - git ref of openvinotoolkit/model_server to pull
#                             export_model.py from (pin to match the served
#                             OVMS image). Defaults to a tag matching 2026.3.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> <llm|vlm|moe|minicpm> [precision]}
MODEL_TYPE=${2:?Usage: $0 <ovms|vllm> <llm|vlm|moe|minicpm> [precision]}
PRECISION=${3:-bf16}
OVMS_EXPORT_MODEL_REF=${OVMS_EXPORT_MODEL_REF:-releases/2026/3}

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "engine must be 'ovms' or 'vllm', got: ${ENGINE}"
[[ "${PRECISION}" == "bf16" || "${PRECISION}" == "int4" ]] || die "precision must be 'bf16' or 'int4', got: ${PRECISION}"

SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
GATED=$(yaml_get "models.${MODEL_TYPE}.gated" "false")

if [[ "${ENGINE}" == "vllm" ]]; then
  require_cmd huggingface-cli

  BF16_REPO=$(yaml_get "models.${MODEL_TYPE}.vllm.precisions.bf16.hf_repo")
  if [[ "${GATED}" == "true" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    die "model ${BF16_REPO} is gated; set HUGGING_FACE_HUB_TOKEN before running"
  fi
  mkdir -p "${MODELS_DIR}"
  log "Downloading ${BF16_REPO} into HF cache for vLLM serving (served as '${SERVED_NAME}')"
  huggingface-cli download "${BF16_REPO}" \
    --cache-dir "${MODELS_DIR}/hf-cache" \
    ${HUGGING_FACE_HUB_TOKEN:+--token "${HUGGING_FACE_HUB_TOKEN}"}
  log "Download complete: ${MODELS_DIR}/hf-cache"

  if [[ "${PRECISION}" == "bf16" ]]; then
    exit 0
  fi

  # --- int4 path: quantize the bf16 checkpoint with torchao, run inside the
  # vLLM serving image so we don't need torch/transformers/torchao on the host.
  require_cmd docker
  VLLM_IMAGE=$(yaml_get "engines.vllm.image")
  TORCHAO_OUT_DIR="${MODELS_DIR}/torchao-int4/${SERVED_NAME}"
  mkdir -p "${TORCHAO_OUT_DIR}"

  if [[ -f "${TORCHAO_OUT_DIR}/.torchao-int4-complete" ]]; then
    log "torchao int4 checkpoint already present at ${TORCHAO_OUT_DIR}, skipping conversion"
    exit 0
  fi

  log "Quantizing ${BF16_REPO} to int4 with torchao (image: ${VLLM_IMAGE}), output: ${TORCHAO_OUT_DIR}"
  docker run --rm \
    -v "${MODELS_DIR}/hf-cache:/root/.cache/huggingface" \
    -v "${TORCHAO_OUT_DIR}:/out" \
    -v "${TOOL_ROOT}/scripts/convert_torchao.py:/opt/convert_torchao.py:ro" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    --entrypoint python3 \
    "${VLLM_IMAGE}" \
    /opt/convert_torchao.py \
      --hf-repo "${BF16_REPO}" \
      --model-type "${MODEL_TYPE}" \
      --output-dir /out \
      ${HUGGING_FACE_HUB_TOKEN:+--hf-token "${HUGGING_FACE_HUB_TOKEN}"}

  log "torchao int4 conversion complete: ${TORCHAO_OUT_DIR}"
  exit 0
fi

# --- OVMS path ---
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

OUT_NAME="${SERVED_NAME}-${PRECISION}"
OUT_DIR="${MODELS_DIR}/${OUT_NAME}"
mkdir -p "${OUT_DIR}"

# Read extra per-model, per-precision export args (weight-format) as a JSON
# array and expand into individual argv entries.
EXTRA_ARGS_JSON=$(yaml_get "models.${MODEL_TYPE}.ovms.precisions.${PRECISION}.export_extra_args" "[]")
mapfile -t EXTRA_ARGS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${EXTRA_ARGS_JSON}")

log "Exporting ${HF_REPO} (precision: ${PRECISION}) to OpenVINO IR at ${OUT_DIR} (extra args: ${EXTRA_ARGS[*]:-none})"
python3 "${EXPORT_SCRIPT}" text_generation \
  --source_model "${HF_REPO}" \
  --model_repository_path "${MODELS_DIR}" \
  --model_name "${OUT_NAME}" \
  --target_device GPU \
  ${HUGGING_FACE_HUB_TOKEN:+--hf_token "${HUGGING_FACE_HUB_TOKEN}"} \
  "${EXTRA_ARGS[@]}"

log "OVMS model export complete: ${OUT_DIR}"
