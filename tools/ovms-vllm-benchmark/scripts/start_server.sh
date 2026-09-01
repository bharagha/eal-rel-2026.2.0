#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Starts the selected model-serving container (OVMS or vLLM) with the
# requested preselected model, using Intel GPU (iGPU) passthrough via
# /dev/dri, and waits until the OpenAI-compatible endpoint is ready.
#
# Usage:
#   scripts/start_server.sh <ovms|vllm> <llm|vlm|moe|minicpm> [precision]
#
# [precision] applies to BOTH engines now (bf16|int4, default bf16):
#   - ovms: serves the precision-scoped IR directory produced by
#           prepare_model.sh (<served_model_name>-<precision>).
#   - vllm: bf16 serves the model's Hugging Face repo id directly; int4
#           serves the local torchao-quantized checkpoint directory produced
#           by prepare_model.sh / convert_torchao.py, with
#           --quantization torchao.
#
# On success prints the base URL of the OpenAI-compatible endpoint to stdout,
# e.g. http://localhost:8000

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> <llm|vlm|moe|minicpm> [precision]}
MODEL_TYPE=${2:?Usage: $0 <ovms|vllm> <llm|vlm|moe|minicpm> [precision]}
PRECISION=${3:-bf16}

require_cmd docker

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "engine must be 'ovms' or 'vllm', got: ${ENGINE}"
[[ "${PRECISION}" == "bf16" || "${PRECISION}" == "int4" ]] || die "precision must be 'bf16' or 'int4', got: ${PRECISION}"

IMAGE=$(yaml_get "engines.${ENGINE}.image")
PORT=$(yaml_get "engines.${ENGINE}.rest_port" "$(yaml_get "engines.${ENGINE}.port")")
HEALTH_PATH=$(yaml_get "engines.${ENGINE}.health_path")
SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
NAME=$(container_name "${ENGINE}" "${PRECISION}")

# Remove any stale container with the same name.
docker rm -f "${NAME}" >/dev/null 2>&1 || true

if [[ "${ENGINE}" == "ovms" ]]; then
  OUT_NAME="${SERVED_NAME}-${PRECISION}"
  log "Starting ${ENGINE} (${IMAGE}) serving '${SERVED_NAME}' (precision: ${PRECISION}) on port ${PORT}"
  docker run -d --name "${NAME}" \
    --device /dev/dri \
    -p "${PORT}:${PORT}" \
    -v "${MODELS_DIR}:/models:ro" \
    "${IMAGE}" \
    --rest_port "${PORT}" \
    --model_path "/models/${OUT_NAME}" \
    --model_name "${SERVED_NAME}" \
    --target_device GPU >/dev/null
else
  log "Starting ${ENGINE} (${IMAGE}) serving '${SERVED_NAME}' (precision: ${PRECISION}) on port ${PORT}"
  VLLM_EXTRA_ARGS_JSON=$(yaml_get "models.${MODEL_TYPE}.vllm.precisions.${PRECISION}.server_extra_args" "[]")
  mapfile -t VLLM_EXTRA_ARGS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${VLLM_EXTRA_ARGS_JSON}")

  if [[ "${PRECISION}" == "int4" ]]; then
    # Serve the local torchao-quantized checkpoint produced by prepare_model.sh.
    TORCHAO_OUT_DIR="${MODELS_DIR}/torchao-int4/${SERVED_NAME}"
    [[ -f "${TORCHAO_OUT_DIR}/.torchao-int4-complete" ]] \
      || die "torchao int4 checkpoint not found at ${TORCHAO_OUT_DIR}; run prepare_model.sh vllm ${MODEL_TYPE} int4 first"
    docker run -d --name "${NAME}" \
      --device /dev/dri \
      -p "${PORT}:${PORT}" \
      -v "${TORCHAO_OUT_DIR}:/model:ro" \
      -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
      "${IMAGE}" \
      --model /model \
      --served-model-name "${SERVED_NAME}" \
      --port "${PORT}" \
      --device xpu \
      "${VLLM_EXTRA_ARGS[@]}" >/dev/null
  else
    HF_REPO=$(yaml_get "models.${MODEL_TYPE}.vllm.precisions.${PRECISION}.hf_repo")
    docker run -d --name "${NAME}" \
      --device /dev/dri \
      -p "${PORT}:${PORT}" \
      -v "${MODELS_DIR}/hf-cache:/root/.cache/huggingface" \
      -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
      "${IMAGE}" \
      --model "${HF_REPO}" \
      --served-model-name "${SERVED_NAME}" \
      --port "${PORT}" \
      --device xpu \
      "${VLLM_EXTRA_ARGS[@]}" >/dev/null
  fi
fi

BASE_URL="http://localhost:${PORT}"
log "Waiting for ${ENGINE} to become ready at ${BASE_URL}${HEALTH_PATH} ..."

READY=0
for _ in $(seq 1 90); do
  if curl -fsS "${BASE_URL}${HEALTH_PATH}" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 10
done

if [[ "${READY}" -ne 1 ]]; then
  log "Server did not become ready in time; recent logs:"
  docker logs --tail 100 "${NAME}" >&2 || true
  die "${ENGINE} server failed to start"
fi

log "${ENGINE} is ready."
echo "${BASE_URL}"
