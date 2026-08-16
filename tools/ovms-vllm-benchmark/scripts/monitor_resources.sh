#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Samples CPU/RAM (via `docker stats`) and Intel GPU utilization/memory
# (via `intel_gpu_top` or, as a fallback, `xpu-smi`) for a running container
# at a fixed interval, writing one JSON object per line (JSONL) until it
# receives SIGTERM.
#
# Usage:
#   scripts/monitor_resources.sh <container_name> <output_jsonl_path> [interval_seconds]
#
# Intended to be run in the background:
#   scripts/monitor_resources.sh my-container /tmp/resources.jsonl 2 &
#   MONITOR_PID=$!
#   ...run benchmark...
#   kill "${MONITOR_PID}"

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

CONTAINER=${1:?Usage: $0 <container_name> <output_jsonl_path> [interval_seconds]}
OUT_FILE=${2:?Usage: $0 <container_name> <output_jsonl_path> [interval_seconds]}
INTERVAL=${3:-2}

mkdir -p "$(dirname "${OUT_FILE}")"
: > "${OUT_FILE}"

GPU_TOOL="none"
if command -v intel_gpu_top >/dev/null 2>&1; then
  GPU_TOOL="intel_gpu_top"
elif command -v xpu-smi >/dev/null 2>&1; then
  GPU_TOOL="xpu-smi"
else
  log "WARNING: neither intel_gpu_top nor xpu-smi found; GPU metrics will be omitted"
fi

sample_gpu() {
  case "${GPU_TOOL}" in
    intel_gpu_top)
      # -J: JSON output, -s: sample duration in ms, -o -: stdout single sample.
      timeout 2 intel_gpu_top -J -s 1000 -o - 2>/dev/null | head -n 1 || echo "null"
      ;;
    xpu-smi)
      xpu-smi dump -d 0 -m 0,1,2,3 -n 1 -j 2>/dev/null || echo "null"
      ;;
    *)
      echo "null"
      ;;
  esac
}

log "Monitoring container '${CONTAINER}' every ${INTERVAL}s (gpu tool: ${GPU_TOOL}) -> ${OUT_FILE}"

trap 'log "Resource monitor stopping."; exit 0' TERM INT

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  docker_stats=$(docker stats --no-stream --format \
    '{{json .}}' "${CONTAINER}" 2>/dev/null || echo "null")
  gpu_sample=$(sample_gpu)

  python3 -c '
import json, sys
ts, docker_stats_raw, gpu_raw = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    docker_stats = json.loads(docker_stats_raw)
except json.JSONDecodeError:
    docker_stats = None
try:
    gpu = json.loads(gpu_raw)
except json.JSONDecodeError:
    gpu = None
print(json.dumps({"timestamp": ts, "docker_stats": docker_stats, "gpu": gpu}))
' "${ts}" "${docker_stats}" "${gpu_sample}" >> "${OUT_FILE}"

  sleep "${INTERVAL}"
done
