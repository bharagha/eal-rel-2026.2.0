#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Stops and removes the benchmark container for the given engine.
#
# Usage:
#   scripts/stop_server.sh <ovms|vllm>

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm>}
NAME=$(container_name "${ENGINE}")

log "Stopping and removing container ${NAME} (if running)"
docker rm -f "${NAME}" >/dev/null 2>&1 || true
log "Done."
