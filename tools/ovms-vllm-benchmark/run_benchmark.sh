#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Interactive entrypoint for the OVMS vs vLLM benchmark tool.
#
# Prompts the user to choose a model-serving engine (OVMS or vLLM), a
# preselected model (LLM / VLM / MoE), and (vLLM only) a serving precision
# (bf16 / int4), then orchestrates the full flow:
#   prepare_model -> start_server -> run_dataset_benchmark (+ resource
#   monitor in the background) -> collect_results -> stop_server.
#
# --precision only applies to the vLLM engine (bf16 default, int4 = w4a16
# quantized checkpoint) and is defined per model in config/models.yaml under
# models.<type>.vllm.precisions.*. It has no effect on OVMS, which continues
# to use its own ovms.export_extra_args (--weight-format) independently.
#
# Usage (interactive):
#   ./run_benchmark.sh
#
# Usage (non-interactive):
#   ./run_benchmark.sh --engine ovms --model-type llm
#   ./run_benchmark.sh --engine vllm --model-type llm --precision int4
#   ./run_benchmark.sh --engine vllm --model-type vlm --num-prompts 20 --keep

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
Usage: $0 [--engine ovms|vllm] [--model-type llm|vlm|moe] [--precision bf16|int4] [--num-prompts N] [--keep]

If --engine/--model-type are omitted, you will be prompted interactively.
  --precision   vLLM-only serving precision (default: bf16). Ignored for OVMS.
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

if [[ "${ENGINE}" == "vllm" ]]; then
  if [[ -z "${PRECISION}" ]]; then
    DEFAULT_PRECISION=$(yaml_get "models.${MODEL_TYPE}.vllm.default_precision" "bf16")
    echo "Which precision would you like to serve with vLLM? [default: ${DEFAULT_PRECISION}]"
    select choice in "bf16" "int4 (w4a16)"; do
      case "${REPLY}" in
        1) PRECISION="bf16"; break ;;
        2) PRECISION="int4"; break ;;
        "") PRECISION="${DEFAULT_PRECISION}"; break ;;
        *) echo "Please choose 1 or 2." ;;
      esac
    done
  fi
  [[ "${PRECISION}" == "bf16" || "${PRECISION}" == "int4" ]] || die "invalid --precision: ${PRECISION}"
  yaml_get "models.${MODEL_TYPE}.vllm.precisions.${PRECISION}.hf_repo" >/dev/null \
    || die "no vllm.precisions.${PRECISION} entry defined for model type '${MODEL_TYPE}' in ${CONFIG_FILE}"
else
  # Precision selection does not apply to OVMS; keep it unset/inert.
  PRECISION="bf16"
fi

NOTES=$(yaml_get "models.${MODEL_TYPE}.notes" "")
[[ -n "${NOTES}" ]] && log "NOTE: ${NOTES}"

if [[ "${ENGINE}" == "vllm" ]]; then
  RUN_ID="${ENGINE}_${MODEL_TYPE}_${PRECISION}_$(date -u +%Y%m%dT%H%M%SZ)"
else
  RUN_ID="${ENGINE}_${MODEL_TYPE}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
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
  --output "${RUN_DIR}/summary.json")
[[ "${ENGINE}" == "vllm" ]] && COLLECT_ARGS+=(--precision "${PRECISION}")
python3 "${TOOL_ROOT}/scripts/collect_results.py" "${COLLECT_ARGS[@]}"

log "Done. Results saved under: ${RUN_DIR}"
