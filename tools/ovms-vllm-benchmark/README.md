<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
# OVMS vs vLLM Benchmark Tool

Interactive benchmarking tool that compares [OpenVINO Model Server (OVMS)](https://github.com/openvinotoolkit/model_server)
and [vLLM](https://github.com/vllm-project/vllm) serving performance on an
**Intel Core Ultra** platform (integrated GPU), using prebuilt, ready-to-use
container images:

- OVMS: `openvino/model_server:2026.3-gpu`
- vLLM: `intel/vllm:0.21.0-xpu`

It downloads/converts a preselected model, starts the chosen serving engine,
runs a serving benchmark against the right dataset for the model type, and
captures both performance (TTFT/TPOT/throughput) and resource usage
(CPU/RAM/GPU) metrics.

> **Platform**: Linux host only (bash scripts + `/dev/dri` GPU passthrough).
> Not supported on Windows.

## Preselected models

| Type | Hugging Face repo | Dataset |
|------|--------------------|---------|
| LLM  | `microsoft/Phi-4-mini-instruct` | ShareGPT-style text prompts |
| VLM  | `Qwen/Qwen3-VL-8B-Instruct` | VQA-style image+prompt dataset |
| MoE  | `google/gemma-4-26B-A4B-it` (gated) | ShareGPT-style text prompts |

See [`config/models.yaml`](config/models.yaml) for exact dataset/model
metadata and per-engine serving arguments.

## Prerequisites

- Docker with Intel GPU (iGPU) drivers installed on the host, `/dev/dri`
  accessible to the current user.
- `python3` (3.10+) with `pyyaml`, `huggingface_hub` CLI (`huggingface-cli`),
  and `pytest` (for running tests) installed.
- Optional but recommended for GPU metrics: `intel_gpu_top` (from
  `intel-gpu-tools`) or `xpu-smi`. If neither is present, GPU utilization
  metrics are omitted with a warning.
- A Hugging Face access token in `HUGGING_FACE_HUB_TOKEN` if you plan to
  benchmark the gated MoE model (`google/gemma-4-26B-A4B-it`). **Never
  hard-code the token** — export it as an environment variable.

## Quick start

```bash
cd tools/ovms-vllm-benchmark
export HUGGING_FACE_HUB_TOKEN=hf_xxx   # only required for gated models
./run_benchmark.sh
```

You will be prompted to choose:
1. The serving engine: **OVMS** or **vLLM**
2. The preselected model: **LLM**, **VLM**, or **MoE**

Non-interactive usage:

```bash
./run_benchmark.sh --engine ovms --model-type llm --num-prompts 50
./run_benchmark.sh --engine vllm --model-type vlm --keep   # leaves the server running afterwards
```

## What it does

1. **Prepare model** (`scripts/prepare_model.sh`) — downloads the model from
   Hugging Face. For OVMS, converts weights to OpenVINO IR using OVMS's own
   `export_model.py` (pinned to match the served image release).
2. **Start server** (`scripts/start_server.sh`) — launches the chosen image
   via `docker run` with `/dev/dri` GPU passthrough, waits for the
   OpenAI-compatible endpoint to become healthy.
3. **Benchmark** (`scripts/run_dataset_benchmark.py`) — runs vLLM's official
   `benchmark_serving.py` against the endpoint using the dataset appropriate
   for the model type, capturing TTFT, TPOT, ITL, and throughput.
4. **Monitor resources** (`scripts/monitor_resources.sh`) — samples
   `docker stats` (CPU/RAM) and `intel_gpu_top`/`xpu-smi` (GPU) in the
   background throughout the benchmark run.
5. **Collect results** (`scripts/collect_results.py`) — merges performance
   and resource metrics into `results/<engine>_<model>_<timestamp>/summary.json`
   and a human-readable `summary.txt`.
6. **Cleanup** — stops and removes the server container unless `--keep` is
   passed, then deactivates and deletes the tool's virtualenv (`venv/`).

## Results layout

```
results/
  ovms_llm_20260101T120000Z/
    benchmark_serving.json   # raw vLLM benchmark_serving.py output
    resources.jsonl          # per-sample docker stats + GPU metrics
    summary.json             # merged performance + resource summary
    summary.txt              # human-readable table
```

## Known limitations / caveats

- `google/gemma-4-26B-A4B-it` is a large MoE model and may exceed the
  integrated GPU's memory on Intel Core Ultra at default settings; the
  config defaults to int4 weight compression for OVMS export, but you may
  need to reduce `--max-model-len` further for vLLM.
- GPU utilization metrics depend on `intel_gpu_top`/`xpu-smi` being
  available and compatible with your driver stack; if unavailable, only
  CPU/RAM metrics are captured.
- NPU targets are not supported in this version — only the integrated GPU
  (iGPU) is exercised.

See [`docs/user-guide/get-started.md`](docs/user-guide/get-started.md) for
more detail.
