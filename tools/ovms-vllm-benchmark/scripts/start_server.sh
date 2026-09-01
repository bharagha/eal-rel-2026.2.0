#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Starts the selected model-serving container (OVMS or vLLM) with the
# requested preselected model, using Intel GPU (iGPU) passthrough via
# /dev/dri, and waits until the OpenAI-compatible endpoint is ready.
#
# Usage:
#   scripts/start_server.sh <ovms|vllm> <llm|vlm|moe> [precision]
#
# [precision] only applies to the vLLM path (bf16|int4, default bf16) and
# selects which entry under models.<type>.vllm.precisions.* is served; it is
# ignored for the OVMS path, which is unaffected by this parameter.
#
# On success prints the base URL of the OpenAI-compatible endpoint to stdout,
# e.g. http://localhost:8000

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> <llm|vlm|moe> [precision]}
MODEL_TYPE=${2:?Usage: $0 <ovms|vllm> <llm|vlm|moe> [precision]}
PRECISION=${3:-bf16}

require_cmd docker

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "engine must be 'ovms' or 'vllm', got: ${ENGINE}"

IMAGE=$(yaml_get "engines.${ENGINE}.image")
PORT=$(yaml_get "engines.${ENGINE}.rest_port" "$(yaml_get "engines.${ENGINE}.port")")
HEALTH_PATH=$(yaml_get "engines.${ENGINE}.health_path")
SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
NAME=$(container_name "${ENGINE}" "${PRECISION}")

# Remove any stale container with the same name.
docker rm -f "${NAME}" >/dev/null 2>&1 || true

if [[ "${ENGINE}" == "ovms" ]]; then
  log "Starting ${ENGINE} (${IMAGE}) serving '${SERVED_NAME}' on port ${PORT}"
  docker run -d --name "${NAME}" \
    --device /dev/dri \
    -p "${PORT}:${PORT}" \
    -v "${MODELS_DIR}:/models:ro" \
    "${IMAGE}" \
    --rest_port "${PORT}" \
    --model_path "/models/${SERVED_NAME}" \
    --model_name "${SERVED_NAME}" \
    --target_device GPU >/dev/null
else
  log "Starting ${ENGINE} (${IMAGE}) serving '${SERVED_NAME}' (precision: ${PRECISION}) on port ${PORT}"
  VLLM_EXTRA_ARGS_JSON=$(yaml_get "models.${MODEL_TYPE}.vllm.precisions.${PRECISION}.server_extra_args" "[]")
  mapfile -t VLLM_EXTRA_ARGS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${VLLM_EXTRA_ARGS_JSON}")
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
