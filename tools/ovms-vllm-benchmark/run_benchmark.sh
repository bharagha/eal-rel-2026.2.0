#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Interactive entrypoint for the OVMS vs vLLM benchmark tool.
#
# Prompts the user to choose a model-serving engine (OVMS or vLLM) and a
# preselected model (LLM / VLM / MoE), then orchestrates the full flow:
#   prepare_model -> start_server -> run_dataset_benchmark (with warm-up +
#   resource metrics) -> collect_results -> stop_server.
#
# Usage (interactive):
#   ./run_benchmark.sh
#
# Usage (non-interactive):
#   ./run_benchmark.sh --engine ovms --model-type llm --warmup-prompts 5
#   ./run_benchmark.sh --engine vllm --model-type vlm --warmup-prompts 5 --num-prompts 20 --keep

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/lib/common.sh"

ENGINE=""
MODEL_TYPE=""
NUM_PROMPTS=5
WARMUP_PROMPTS=0
KEEP=0

usage() {
  cat <<EOF
Usage: $0 [--engine ovms|vllm] [--model-type llm|vlm|moe] [--num-prompts N] [--warmup-prompts N] [--keep]

If --engine/--model-type are omitted, you will be prompted interactively.
  --warmup-prompts N  Send N requests before resource collection and measurement.
  --keep    Leave the server container running after the benchmark completes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE=$2; shift 2 ;;
    --model-type) MODEL_TYPE=$2; shift 2 ;;
    --num-prompts) NUM_PROMPTS=$2; shift 2 ;;
    --warmup-prompts) WARMUP_PROMPTS=$2; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if [[ -z "${ENGINE}" ]]; then
  echo "Which model serving engine would you like to benchmark?"
  select choice in "OVMS" "vLLM"; do
    case "${choice}" in
      OVMS) ENGINE="ovms"; break ;;
      vLLM) ENGINE="vllm"; break ;;
      *) echo "Please choose 1 or 2." ;;
    esac
  done
fi

if [[ -z "${MODEL_TYPE}" ]]; then
  echo "Which preselected model would you like to benchmark?"
  select choice in "LLM (microsoft/Phi-4-mini-instruct)" "VLM (Qwen/Qwen3-VL-8B-Instruct)" "MoE (google/gemma-4-26B-A4B-it)"; do
    case "${REPLY}" in
      1) MODEL_TYPE="llm"; break ;;
      2) MODEL_TYPE="vlm"; break ;;
      3) MODEL_TYPE="moe"; break ;;
      *) echo "Please choose 1, 2, or 3." ;;
    esac
  done
fi

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "invalid --engine: ${ENGINE}"
[[ "${MODEL_TYPE}" == "llm" || "${MODEL_TYPE}" == "vlm" || "${MODEL_TYPE}" == "moe" ]] || die "invalid --model-type: ${MODEL_TYPE}"
[[ "${NUM_PROMPTS}" =~ ^[0-9]+$ ]] || die "--num-prompts must be a non-negative integer"
[[ "${WARMUP_PROMPTS}" =~ ^[0-9]+$ ]] || die "--warmup-prompts must be a non-negative integer"

NOTES=$(yaml_get "models.${MODEL_TYPE}.notes" "")
[[ -n "${NOTES}" ]] && log "NOTE: ${NOTES}"

RUN_ID="${ENGINE}_${MODEL_TYPE}_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
log "Run directory: ${RUN_DIR}"

cleanup() {
  if [[ "${KEEP}" -eq 0 ]]; then
    "${TOOL_ROOT}/scripts/stop_server.sh" "${ENGINE}" || true
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]] && command -v deactivate >/dev/null 2>&1; then
    log "Deactivating virtualenv"
    deactivate || true
  fi
  # if [[ "${KEEP}" -eq 0 && -d "${TOOL_ROOT}/venv" ]]; then
  #   log "Removing virtualenv (${TOOL_ROOT}/venv)"
  #   rm -rf "${TOOL_ROOT}/venv" || true
  # fi
}
trap cleanup EXIT

log "Step 1/4: preparing model (download${ENGINE:+/convert for ovms})"
"${TOOL_ROOT}/scripts/prepare_model.sh" "${ENGINE}" "${MODEL_TYPE}"

log "Step 2/4: starting ${ENGINE} server"
BASE_URL=$("${TOOL_ROOT}/scripts/start_server.sh" "${ENGINE}" "${MODEL_TYPE}" | tail -n 1)
log "Server endpoint: ${BASE_URL}"

SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
BENCH_JSON="${RUN_DIR}/benchmark_serving.json"

log "Step 3/4: warming up and running dataset benchmark (num_prompts=${NUM_PROMPTS})"
python3 "${TOOL_ROOT}/scripts/run_dataset_benchmark.py" \
  --base-url "${BASE_URL}" \
  --model-type "${MODEL_TYPE}" \
  --served-model-name "${SERVED_NAME}" \
  --config "${CONFIG_FILE}" \
  --output "${BENCH_JSON}" \
  --report-dir "${RUN_DIR}" \
  --num-prompts "${NUM_PROMPTS}" \
  --warmup-prompts "${WARMUP_PROMPTS}"

log "Step 4/4: collecting results"
python3 "${TOOL_ROOT}/scripts/collect_results.py" \
  --engine "${ENGINE}" \
  --model-type "${MODEL_TYPE}" \
  --benchmark-json "${BENCH_JSON}" \
  --config "${CONFIG_FILE}" \
  --output "${RUN_DIR}/summary.json"

log "Done. Results saved under: ${RUN_DIR}"
