#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stops and removes the benchmark container for the given engine.
#
# Usage:
#   scripts/stop_server.sh <ovms|vllm> [precision]
#
# [precision] must match what was passed to start_server.sh so the correct
# container name is resolved (applies to both engines now).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> [precision]}
PRECISION=${2:-bf16}
NAME=$(container_name "${ENGINE}" "${PRECISION}")

log "Stopping and removing container ${NAME} (if running)"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
log "Done."
