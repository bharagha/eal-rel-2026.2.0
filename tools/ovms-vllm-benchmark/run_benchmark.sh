#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Interactive entrypoint for the OVMS vs vLLM benchmark tool.
#
# Prompts the user to choose a model-serving engine (OVMS or vLLM), a
# preselected model (LLM / VLM / MoE / MiniCPM), and a serving precision
# (bf16 / int4), then orchestrates the full flow:
#   prepare_model -> start_server -> run_dataset_benchmark (+ resource
#   monitor in the background) -> collect_results -> stop_server.
#
# --precision now applies to BOTH engines (bf16 default, int4) and is
# defined per model in config/models.yaml under models.<type>.<engine>.precisions.*.
#   - vllm: bf16 serves the model's original Hugging Face repo; int4 serves a
#           torchao-quantized checkpoint produced on the fly by
#           scripts/convert_torchao.py.
#   - ovms: bf16/int4 select the --weight-format used by export_model.py
#           (fp16 / int4 respectively) via models.<type>.ovms.precisions.*.
#
# Usage (interactive):
#   ./run_benchmark.sh
#
# Usage (non-interactive):
#   ./run_benchmark.sh --engine ovms --model-type llm
#   ./run_benchmark.sh --engine ovms --model-type llm --precision int4
#   ./run_benchmark.sh --engine vllm --model-type llm --precision int4
#   ./run_benchmark.sh --engine vllm --model-type vlm --num-prompts 20 --keep
#   ./run_benchmark.sh --engine vllm --model-type minicpm --precision int4

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/lib/common.sh"

ENGINE=""
MODEL_TYPE=""
PRECISION=""
NUM_PROMPTS=50
KEEP=0
MONITOR_INTERVAL=2

usage() {
  cat <<EOF
Usage: $0 [--engine ovms|vllm] [--model-type llm|vlm|moe|minicpm] [--precision bf16|int4] [--num-prompts N] [--keep]

If --engine/--model-type are omitted, you will be prompted interactively.
  --precision   Serving precision (default: bf16), supported by both engines.
  --keep        Leave the server container running after the benchmark completes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE=$2; shift 2 ;;
    --model-type) MODEL_TYPE=$2; shift 2 ;;
    --precision) PRECISION=$2; shift 2 ;;
    --num-prompts) NUM_PROMPTS=$2; shift 2 ;;
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
  select choice in "LLM (microsoft/Phi-4-mini-instruct)" "VLM (Qwen/Qwen3-VL-8B-Instruct)" "MoE (google/gemma-4-26B-A4B-it)" "MiniCPM (openbmb/MiniCPM-V-4_5)"; do
    case "${REPLY}" in
      1) MODEL_TYPE="llm"; break ;;
      2) MODEL_TYPE="vlm"; break ;;
      3) MODEL_TYPE="moe"; break ;;
      4) MODEL_TYPE="minicpm"; break ;;
      *) echo "Please choose 1, 2, 3, or 4." ;;
    esac
  done
fi

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "invalid --engine: ${ENGINE}"
[[ "${MODEL_TYPE}" == "llm" || "${MODEL_TYPE}" == "vlm" || "${MODEL_TYPE}" == "moe" || "${MODEL_TYPE}" == "minicpm" ]] || die "invalid --model-type: ${MODEL_TYPE}"

if [[ -z "${PRECISION}" ]]; then
  DEFAULT_PRECISION=$(yaml_get "models.${MODEL_TYPE}.${ENGINE}.default_precision" "bf16")
  echo "Which precision would you like to serve with ${ENGINE}? [default: ${DEFAULT_PRECISION}]"
  select choice in "bf16" "int4"; do
    case "${REPLY}" in
      1) PRECISION="bf16"; break ;;
      2) PRECISION="int4"; break ;;
      "") PRECISION="${DEFAULT_PRECISION}"; break ;;
      *) echo "Please choose 1 or 2." ;;
    esac
  done
fi
[[ "${PRECISION}" == "bf16" || "${PRECISION}" == "int4" ]] || die "invalid --precision: ${PRECISION}"

if [[ "${ENGINE}" == "vllm" ]]; then
  yaml_get "models.${MODEL_TYPE}.vllm.precisions.${PRECISION}" >/dev/null 2>&1 \
    || die "no vllm.precisions.${PRECISION} entry defined for model type '${MODEL_TYPE}' in ${CONFIG_FILE}"
else
  yaml_get "models.${MODEL_TYPE}.ovms.precisions.${PRECISION}" >/dev/null 2>&1 \
    || die "no ovms.precisions.${PRECISION} entry defined for model type '${MODEL_TYPE}' in ${CONFIG_FILE}"
fi

NOTES=$(yaml_get "models.${MODEL_TYPE}.notes" "")
[[ -n "${NOTES}" ]] && log "NOTE: ${NOTES}"

RUN_ID="${ENGINE}_${MODEL_TYPE}_${PRECISION}_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
log "Run directory: ${RUN_DIR}"

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ "${KEEP}" -eq 0 ]]; then
    "${TOOL_ROOT}/scripts/stop_server.sh" "${ENGINE}" "${PRECISION}" || true
  fi
}
trap cleanup EXIT

log "Step 1/4: preparing model (download${ENGINE:+/convert for ovms})"
"${TOOL_ROOT}/scripts/prepare_model.sh" "${ENGINE}" "${MODEL_TYPE}" "${PRECISION}"

log "Step 2/4: starting ${ENGINE} server"
BASE_URL=$("${TOOL_ROOT}/scripts/start_server.sh" "${ENGINE}" "${MODEL_TYPE}" "${PRECISION}" | tail -n 1)
log "Server endpoint: ${BASE_URL}"

CONTAINER=$(container_name "${ENGINE}" "${PRECISION}")
RESOURCES_FILE="${RUN_DIR}/resources.jsonl"
"${TOOL_ROOT}/scripts/monitor_resources.sh" "${CONTAINER}" "${RESOURCES_FILE}" "${MONITOR_INTERVAL}" &
MONITOR_PID=$!
log "Resource monitor started (pid ${MONITOR_PID})"

SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
BENCH_JSON="${RUN_DIR}/benchmark_serving.json"

log "Step 3/4: running dataset benchmark (num_prompts=${NUM_PROMPTS})"
python3 "${TOOL_ROOT}/scripts/run_dataset_benchmark.py" \
  --base-url "${BASE_URL}" \
  --model-type "${MODEL_TYPE}" \
  --served-model-name "${SERVED_NAME}" \
  --config "${CONFIG_FILE}" \
  --output "${BENCH_JSON}" \
  --num-prompts "${NUM_PROMPTS}"

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""

log "Step 4/4: collecting results"
COLLECT_ARGS=(--engine "${ENGINE}" --model-type "${MODEL_TYPE}" \
  --benchmark-json "${BENCH_JSON}" --resources-jsonl "${RESOURCES_FILE}" \
  --output "${RUN_DIR}/summary.json" --precision "${PRECISION}")
python3 "${TOOL_ROOT}/scripts/collect_results.py" "${COLLECT_ARGS[@]}"

log "Done. Results saved under: ${RUN_DIR}"
