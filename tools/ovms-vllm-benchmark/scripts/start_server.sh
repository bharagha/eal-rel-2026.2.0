#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Starts the selected model-serving container (OVMS or vLLM) with the
# requested preselected model, using Intel GPU (iGPU) passthrough via
# /dev/dri, and waits until the OpenAI-compatible endpoint is ready.
#
# Usage:
#   scripts/start_server.sh <ovms|vllm> <llm|vlm|moe>
#
# On success prints the base URL of the OpenAI-compatible endpoint to stdout,
# e.g. http://localhost:8000

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> <llm|vlm|moe>}
MODEL_TYPE=${2:?Usage: $0 <ovms|vllm> <llm|vlm|moe>}

require_cmd docker

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "engine must be 'ovms' or 'vllm', got: ${ENGINE}"

IMAGE=$(yaml_get "engines.${ENGINE}.image")
PORT=$(yaml_get "engines.${ENGINE}.rest_port" "$(yaml_get "engines.${ENGINE}.port")")
HEALTH_PATH=$(yaml_get "engines.${ENGINE}.health_path")
SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
NAME=$(container_name "${ENGINE}")

# Remove any stale container with the same name.
docker rm -f "${NAME}" >/dev/null 2>&1 || true

log "Starting ${ENGINE} (${IMAGE}) serving '${SERVED_NAME}' on port ${PORT}"

# Forward host proxy settings into the container so it can reach the network.
PROXY_ARGS=()
for _var in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
  if [[ -n "${!_var:-}" ]]; then
    PROXY_ARGS+=(-e "${_var}=${!_var}")
  fi
done

if [[ "${ENGINE}" == "ovms" ]]; then
  RENDER_GID=$(stat -c "%g" /dev/dri/render* 2>/dev/null | head -n 1)
  docker run -d --name "${NAME}" \
    --user "$(id -u):$(id -g)" \
    --device /dev/dri \
    ${RENDER_GID:+--group-add="${RENDER_GID}"} \
    -p "${PORT}:${PORT}" \
    -v "${MODELS_DIR}:/models" \
    "${PROXY_ARGS[@]}" \
    "${IMAGE}" \
    --config_path /models/config.json \
    --rest_port "${PORT}" >/dev/null
else
  VLLM_EXTRA_ARGS_JSON=$(yaml_get "models.${MODEL_TYPE}.vllm.server_extra_args" "[]")
  mapfile -t VLLM_EXTRA_ARGS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${VLLM_EXTRA_ARGS_JSON}")
  HF_REPO=$(yaml_get "models.${MODEL_TYPE}.hf_repo")

  # oneCCL's ze_fd_manager enumerates GPU device fds by scanning /dev/dri
  # (including the by-path/ symlink directory). Passing only `--device
  # /dev/dri` maps the char device nodes but not the by-path directory, so
  # oneCCL fails with "opendir failed: could not open device directory" during
  # XPU init. Bind-mount by-path and add the render group so the XPU is usable.
  RENDER_GID=$(stat -c "%g" /dev/dri/render* 2>/dev/null | head -n 1)
  docker run -d --name "${NAME}" \
    --device /dev/dri \
    ${RENDER_GID:+--group-add="${RENDER_GID}"} \
    -v /dev/dri/by-path:/dev/dri/by-path:ro \
    --shm-size=8g \
    -p "${PORT}:${PORT}" \
    -v "${MODELS_DIR}/hf-cache:/root/.cache/huggingface" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    "${PROXY_ARGS[@]}" \
    "${IMAGE}" \
    vllm serve "${HF_REPO}" \
    --served-model-name "${SERVED_NAME}" \
    --port "${PORT}" \
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
