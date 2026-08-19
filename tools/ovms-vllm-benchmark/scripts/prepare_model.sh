#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Downloads (and, for OVMS, converts to OpenVINO IR) a preselected model
# ready for serving by either engine.
#
# Usage:
#   scripts/prepare_model.sh <ovms|vllm> <llm|vlm|moe>
#
# Env vars:
#   HUGGING_FACE_HUB_TOKEN  - required for gated models (e.g. the MoE entry).
#   MODELS_DIR              - override local model cache directory.
#   OVMS_EXPORT_MODEL_REF   - git ref of openvinotoolkit/model_server to pull
#                             export_model.py from (pin to match the served
#                             OVMS image). Defaults to a tag matching 2026.3.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ENGINE=${1:?Usage: $0 <ovms|vllm> <llm|vlm|moe>}
MODEL_TYPE=${2:?Usage: $0 <ovms|vllm> <llm|vlm|moe>}
OVMS_EXPORT_MODEL_REF=${OVMS_EXPORT_MODEL_REF:-releases/2026/3}

[[ "${ENGINE}" == "ovms" || "${ENGINE}" == "vllm" ]] || die "engine must be 'ovms' or 'vllm', got: ${ENGINE}"

HF_REPO=$(yaml_get "models.${MODEL_TYPE}.hf_repo")
SERVED_NAME=$(yaml_get "models.${MODEL_TYPE}.served_model_name")
GATED=$(yaml_get "models.${MODEL_TYPE}.gated" "false")

if [[ "${GATED}" == "true" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  die "model ${HF_REPO} is gated; set HUGGING_FACE_HUB_TOKEN before running"
fi

mkdir -p "${MODELS_DIR}"

if [[ "${ENGINE}" == "vllm" ]]; then
  require_cmd hf
  log "Downloading ${HF_REPO} into HF cache for vLLM serving (served as '${SERVED_NAME}')"
  hf download "${HF_REPO}" \
    --cache-dir "${MODELS_DIR}/hf-cache" \
    ${HUGGING_FACE_HUB_TOKEN:+--token "${HUGGING_FACE_HUB_TOKEN}"}
  log "Download complete: ${MODELS_DIR}/hf-cache"
  exit 0
fi

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

# The upstream export_model.py hardcodes `--trust-remote-code` on the
# text_generation optimum-cli export. For models whose type is natively
# supported by the installed transformers (e.g. Phi-4-mini / phi3), optimum
# disables remote code during export but then reloads the exported model with
# trust_remote_code=True for weight compression, causing it to look for a
# `configuration_*.py` that was never copied into the temp export dir
# (OSError: ... does not have a file named configuration_phi3.py). Drop that
# flag from the text_generation command so int8 export succeeds. Idempotent.
if grep -q -- "--weight-format {} {} --trust-remote-code {}\".format(source_model" "${EXPORT_SCRIPT}"; then
  log "Patching export_model.py: removing --trust-remote-code from text_generation export command"
  sed -i 's/--weight-format {} {} --trust-remote-code {}"\.format(source_model/--weight-format {} {} {}".format(source_model/' "${EXPORT_SCRIPT}"
fi

# Some models cannot be exported with the modern OVMS 2026.3 export stack
# (transformers 5.x / huggingface-hub 1.x). For those, config/models.yaml sets
# `ovms.legacy_export.{transformers,huggingface_hub}` and the export runs in an
# isolated, pinned virtualenv so the main venv is left untouched.
LEGACY_TF=$(yaml_get "models.${MODEL_TYPE}.ovms.legacy_export.transformers" "")
LEGACY_HFHUB=$(yaml_get "models.${MODEL_TYPE}.ovms.legacy_export.huggingface_hub" "")

if [[ -n "${LEGACY_TF}" ]]; then
  LEGACY_VENV="${EXPORT_SCRIPT_DIR}/venv-legacy"
  MARKER="${LEGACY_VENV}/.pins-${LEGACY_TF}-${LEGACY_HFHUB//[<>=]/_}"
  if [[ ! -f "${MARKER}" ]]; then
    log "Setting up isolated export venv for ${SERVED_NAME} (transformers==${LEGACY_TF}, huggingface-hub${LEGACY_HFHUB})"
    [[ -d "${LEGACY_VENV}" ]] || python3 -m venv "${LEGACY_VENV}"
    "${LEGACY_VENV}/bin/python" -m pip install --quiet --upgrade pip
    if [[ -f "${EXPORT_SCRIPT_DIR}/requirements.txt" ]]; then
      "${LEGACY_VENV}/bin/python" -m pip install --quiet -r "${EXPORT_SCRIPT_DIR}/requirements.txt"
    fi
    # Pin the older transformers/huggingface-hub required by this model's
    # remote code and optimum-intel export config. Installed together so their
    # transitive constraints (tokenizers, safetensors) resolve consistently.
    "${LEGACY_VENV}/bin/python" -m pip install --quiet \
      "transformers==${LEGACY_TF}" "huggingface-hub${LEGACY_HFHUB}"
    touch "${MARKER}"
  fi
  PYBIN="${LEGACY_VENV}/bin/python"
  EXPORT_BIN_DIR="${LEGACY_VENV}/bin"
else
  PYBIN="python3"
  EXPORT_BIN_DIR=""
  if [[ -f "${EXPORT_SCRIPT_DIR}/requirements.txt" ]]; then
    pip3 install --quiet -r "${EXPORT_SCRIPT_DIR}/requirements.txt"
  fi

  # The pinned optimum-intel dev build declares `transformers<5.1` in its package
  # metadata, but its OpenVINO export code (the gemma4 export configs) actually
  # requires transformers >= 5.5, where the `gemma4` model_type was introduced.
  # Installing the export requirements above therefore pulls transformers 5.0.x,
  # which cannot load the MoE model config and fails with `KeyError: 'gemma4'`.
  # Force a gemma4-capable transformers with --no-deps so the stale upper-bound
  # pin does not downgrade it back. Idempotent.
  if ! python3 -c "import transformers; from packaging.version import parse as v; import sys; sys.exit(0 if v(transformers.__version__) >= v('5.5.0') else 1)" 2>/dev/null; then
    log "Installing gemma4-capable transformers (>=5.5) required by optimum-intel's export configs"
    pip3 install --quiet --no-deps "transformers==5.5.0"
  fi
fi

OUT_DIR="${MODELS_DIR}/${SERVED_NAME}"

# A previous aborted run (or an OVMS auto-pull) may leave an incomplete export
# containing only graph.pbtxt. export_model.py skips conversion when the output
# directory already exists, so remove any export that lacks the IR weights to
# force a clean re-export. Text-only models produce openvino_model.xml; VLMs
# produce openvino_language_model.xml (plus vision submodels).
if [[ -d "${OUT_DIR}" && ! -f "${OUT_DIR}/openvino_model.xml" && ! -f "${OUT_DIR}/openvino_language_model.xml" ]]; then
  log "Removing incomplete model export at ${OUT_DIR}"
  rm -rf "${OUT_DIR}"
fi

# Read extra per-model export args (e.g. weight-format) as a JSON array and
# expand into individual argv entries.
EXTRA_ARGS_JSON=$(yaml_get "models.${MODEL_TYPE}.ovms.export_extra_args" "[]")
mapfile -t EXTRA_ARGS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "${EXTRA_ARGS_JSON}")

log "Exporting ${HF_REPO} to OpenVINO IR at ${OUT_DIR} (extra args: ${EXTRA_ARGS[*]:-none})"
# export_model.py shells out to `optimum-cli` / `convert_tokenizer` via
# os.system(), which resolve from PATH. When a legacy export venv is used, put
# its bin dir (and its VIRTUAL_ENV) first so those subprocesses run under the
# pinned transformers/huggingface-hub rather than the activated main venv.
if [[ -n "${EXPORT_BIN_DIR}" ]]; then
  PATH="${EXPORT_BIN_DIR}:${PATH}" VIRTUAL_ENV="${LEGACY_VENV}" "${PYBIN}" "${EXPORT_SCRIPT}" text_generation \
    --source_model "${HF_REPO}" \
    --model_repository_path "${MODELS_DIR}" \
    --model_name "${SERVED_NAME}" \
    --config_file_path "${MODELS_DIR}/config.json" \
    --target_device GPU \
    ${HUGGING_FACE_HUB_TOKEN:+--hf_token "${HUGGING_FACE_HUB_TOKEN}"} \
    "${EXTRA_ARGS[@]}"
else
  "${PYBIN}" "${EXPORT_SCRIPT}" text_generation \
    --source_model "${HF_REPO}" \
    --model_repository_path "${MODELS_DIR}" \
    --model_name "${SERVED_NAME}" \
    --config_file_path "${MODELS_DIR}/config.json" \
    --target_device GPU \
    ${HUGGING_FACE_HUB_TOKEN:+--hf_token "${HUGGING_FACE_HUB_TOKEN}"} \
    "${EXTRA_ARGS[@]}"
fi

log "OVMS model export complete: ${OUT_DIR}"
