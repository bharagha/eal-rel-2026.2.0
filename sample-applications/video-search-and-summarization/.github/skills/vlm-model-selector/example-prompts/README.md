<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Example prompts: vlm-model-selector

Each file is a ready-to-use prompt that can be passed via `@file`.
Edit the hardware and workload values to match the deployment.

| File | Prompt |
|---|---|
| [01-retail-open-search.md](./01-retail-open-search.md) | Compare current VLMs for retail video alerts on a Core Ultra iGPU |
| [02-existing-shortlist.md](./02-existing-shortlist.md) | Evaluate an exact checkpoint for document QA without expanding scope |
| [03-npu-defect-description.md](./03-npu-defect-description.md) | Select an NPU-first model for offline visual defect descriptions |
| [04-arc-video-review.md](./04-arc-video-review.md) | Compare discrete-GPU models for recorded video review |
| [05-two-stage-incident-analysis.md](./05-two-stage-incident-analysis.md) | Select a small/large VLM cascade with explicit concurrency and latency |
| [06-clarify-selection-requirements.md](./06-clarify-selection-requirements.md) | Start with incomplete requirements and choose the search approach |

These are interactive usage examples, not fixed model recommendations.
They may require current Hub, license, leaderboard, and OpenVINO lookups.
Unavailable sources must be reported rather than replaced with invented
results. A deployment being offline does not prevent online research or
preparing model assets before deployment.

The separate [evaluation cases](../evals/evals.json) use self-contained
prompts to check workflow behavior without downloading models or requiring
network access. Their synthetic fixtures are not real model evidence.
Passing those cases does not validate live discovery or inference performance.
