<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
# Get Started: OVMS vs vLLM Benchmark Tool

## Overview

This tool benchmarks [OpenVINO Model Server (OVMS)](https://github.com/openvinotoolkit/model_server)
against [vLLM](https://github.com/vllm-project/vllm) on an Intel Core Ultra
platform, using the prebuilt images `openvino/model_server:2026.3-gpu` and
`intel/vllm:0.21.0-xpu`. It measures request-level latency (TTFT, TPOT) and
throughput via vLLM's official `benchmark_serving.py`, and captures CPU/RAM/
GPU utilization for the serving container throughout the run.

## System requirements

- Linux host with Intel Core Ultra (integrated GPU) and up-to-date Intel GPU
  drivers (`/dev/dri/renderD*` present and accessible).
- Docker Engine (rootless or with the invoking user in the `docker` group).
- Free disk space for model weights (multi-GB per model; the MoE model is
  significantly larger).
- Python 3.10+ with `pip install pyyaml pytest` and the Hugging Face CLI
  (`pip install huggingface_hub[cli]`).
- Optional: `intel-gpu-tools` (`intel_gpu_top`) or `xpu-smi` for GPU
  utilization sampling.

## Step-by-step

1. **Clone/enter the tool directory**

   ```bash
   cd tools/ovms-vllm-benchmark
   ```

2. **Set your Hugging Face token** (only required for the gated MoE model)

   ```bash
   export HUGGING_FACE_HUB_TOKEN=hf_xxx
   ```

   Never commit this value to source control; keep it as an environment
   variable or in a secret manager.

3. **Run the benchmark**

   Interactively:

   ```bash
   ./run_benchmark.sh
   ```

   You'll be asked:
   - *"Which model serving engine would you like to benchmark?"* → OVMS or vLLM
   - *"Which preselected model would you like to benchmark?"* → LLM, VLM, or MoE

   Or non-interactively:

   ```bash
   ./run_benchmark.sh --engine ovms --model-type llm --num-prompts 50
   ```

4. **Review results**

   Results are written to `results/<engine>_<model-type>_<UTC-timestamp>/`:
   - `benchmark_serving.json` — raw vLLM benchmark output (TTFT, TPOT, ITL,
     throughput, per-request details)
   - `resources.jsonl` — time-series of CPU/RAM (`docker stats`) and GPU
     utilization samples
   - `summary.json` / `summary.txt` — merged, human-readable summary

5. **Compare runs**

   Run the same `--model-type` against both `--engine ovms` and
   `--engine vllm`, then diff the two `summary.json`/`summary.txt` files to
   compare TTFT/TPOT and resource usage side by side.

## Customizing

- Edit [`config/models.yaml`](../../config/models.yaml) to change the
  preselected Hugging Face model IDs, datasets, or per-engine serving
  arguments (e.g., `--max-model-len`, weight compression format).
- Override `MODELS_DIR`/`RESULTS_DIR` environment variables to relocate the
  local model cache / results output.
- Pass `--num-prompts` to control benchmark load; pass `--keep` to leave the
  server container running after the benchmark for manual inspection.

## Troubleshooting

- **Server never becomes ready**: check `docker logs <container>` — the
  script prints the last 100 lines automatically on timeout. Common causes:
  insufficient GPU memory for the selected model, or missing
  `HUGGING_FACE_HUB_TOKEN` for gated models.
- **GPU metrics show `n/a`**: install `intel-gpu-tools` or `xpu-smi`, or
  check driver compatibility; the tool degrades gracefully without them.
- **`/dev/dri` permission denied**: ensure the invoking user/container has
  access to the render node (commonly the `render` group on the host).
