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

| Type    | Hugging Face repo | Dataset | Precisions (both engines) |
|---------|--------------------|---------|----------------------------|
| LLM     | `microsoft/Phi-4-mini-instruct` | ShareGPT-style text prompts | `bf16`, `int4` |
| VLM     | `Qwen/Qwen3-VL-8B-Instruct` | VQA-style image+prompt dataset | `bf16`, `int4` |
| MoE     | `google/gemma-4-26B-A4B-it` (gated) | ShareGPT-style text prompts | `bf16`, `int4` |
| MiniCPM | `openbmb/MiniCPM-V-4_5` | VQA-style image+prompt dataset | `bf16`, `int4` |

See [`config/models.yaml`](config/models.yaml) for exact dataset/model
metadata and per-engine, per-precision serving arguments.

- **OVMS**: `bf16` exports OpenVINO IR with `--weight-format fp16` (the
  closest full-precision option `export_model.py` supports; there is no
  literal "bf16" weight-format); `int4` exports with `--weight-format int4`.
  Each precision is exported to its own directory
  (`<served_model_name>-<precision>`) so both can coexist locally.
- **vLLM**: `bf16` serves the model's original Hugging Face repo directly;
  `int4` quantizes the downloaded bf16 checkpoint on the fly using
  [torchao](https://github.com/pytorch/ao)'s `Int4WeightOnlyConfig`
  (weight-only int4), caches the converted checkpoint locally, and serves it
  with `--quantization torchao`. No separate pre-quantized Hugging Face repo
  is required.

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
2. The preselected model: **LLM**, **VLM**, **MoE**, or **MiniCPM**
3. The serving precision: **bf16** (default) or **int4** — supported by
   both engines

Non-interactive usage:

```bash
./run_benchmark.sh --engine ovms --model-type llm --num-prompts 50
./run_benchmark.sh --engine ovms --model-type llm --precision int4
./run_benchmark.sh --engine vllm --model-type llm --precision int4
./run_benchmark.sh --engine vllm --model-type vlm --keep   # leaves the server running afterwards
./run_benchmark.sh --engine vllm --model-type minicpm --precision int4
```

### Precision (bf16 / int4, both engines)

`--precision bf16|int4` selects which entry under
`models.<type>.<engine>.precisions.*` in `config/models.yaml` is served.
Every preselected model defines both precisions for both engines:

- **OVMS**: `bf16` → `--weight-format fp16` IR export; `int4` → native
  `--weight-format int4` IR export (OpenVINO weight compression). Exported
  once per precision and cached under `model-cache/<served_model_name>-<precision>`.
- **vLLM**: `bf16` serves the original Hugging Face repo as-is; `int4`
  quantizes that same bf16 checkpoint on first use with torchao
  (`Int4WeightOnlyConfig`, weight-only int4), caching the converted
  checkpoint under `model-cache/torchao-int4/<served_model_name>`, and
  serves it with `--quantization torchao`.

## What it does

1. **Prepare model** (`scripts/prepare_model.sh`) — downloads the model from
   Hugging Face. For OVMS, converts weights to OpenVINO IR at the selected
   precision using OVMS's own `export_model.py` (pinned to match the served
   image release). For vLLM `int4`, additionally quantizes the downloaded
   bf16 checkpoint to int4 with torchao (`scripts/convert_torchao.py`, run
   inside the `intel/vllm` serving image).
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
   passed.

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
  integrated GPU's memory on Intel Core Ultra at default settings; use
  `--precision int4` (OVMS: native int4 IR export; vLLM: torchao int4
  weight-only quantization) and/or reduce `--max-model-len` further.
- The vLLM torchao int4 conversion runs on first use per model and can take
  several minutes and significant host RAM/disk (a full bf16 checkpoint plus
  the quantized copy are both materialized); subsequent runs reuse the
  cached checkpoint under `model-cache/torchao-int4/<served_model_name>`.
- GPU utilization metrics depend on `intel_gpu_top`/`xpu-smi` being
  available and compatible with your driver stack; if unavailable, only
  CPU/RAM metrics are captured.
- NPU targets are not supported in this version — only the integrated GPU
  (iGPU) is exercised.

See [`docs/user-guide/get-started.md`](docs/user-guide/get-started.md) for
more detail.
