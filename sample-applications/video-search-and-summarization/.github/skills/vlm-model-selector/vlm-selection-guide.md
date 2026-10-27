<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
Adapted from the vlm-model-selector selection procedure.
-->

# Selecting a Vision-Language Model for Intel Hardware

## Purpose and scope

This guide describes how to select a vision-language model (VLM) for an image or video application running on Intel Core Ultra NPU/iGPU or Intel Arc/Arc Pro discrete GPUs.

The selection has four parts:

1. Match the model's capabilities to the application.
2. Check relevant accuracy evidence.
3. Verify runtime compatibility and hardware fit.
4. Confirm that the license permits the deployment.

The result should be a ranked comparison, a primary model, a fallback, and a list of validation work still required. A benchmark score or a successful export alone is not enough to approve a model for deployment.

This procedure covers model selection and initial deployment checks. It does not cover training, fine-tuning, or text-only LLM selection.

## 1. Define the workload

Write down the requirements before searching for models.

| Requirement | What to record |
|---|---|
| Task | The observations, questions, or events the application must handle |
| Input | Single image, multiple images, sampled live video, or recorded clips |
| Output | Free-form text, JSON, bounding boxes, or a fixed set of labels |
| Hardware | Exact processor/GPU, available RAM or VRAM, OS, and intended inference device |
| Accuracy | Task-specific acceptance criteria and acceptable failure modes |
| Latency | Where the clock starts and stops; required percentile or hard deadline |
| Throughput | Requests per second, camera count, sampling rate, and concurrent requests |
| Deployment | Commercial or research use, offline operation, and data-residency requirements |
| Language | Required input and output languages |
| Precision | Whether INT4/INT8 quantization is acceptable |

For video, also record the resolution, frames per request, clip duration, and expected incident rate. Distinguish steady-state traffic from bursts.

If hardware details are not yet available, use an explicit planning assumption, such as a Core Ultra Series 2/3 system with 16-32 GB shared RAM. Do not treat that assumption as verified hardware capacity.

### Confirm that a VLM is needed

Use a VLM when the task needs visual question answering, descriptions, document understanding, or reasoning over images and video.

If the output is only a count, class, mask, or location, first evaluate a detector, classifier, or segmentation model. A hybrid pipeline may be more appropriate: specialist vision models produce detections and tracks, and a VLM explains selected events.

Do not assume that a general-purpose VLM provides reliable pixel-level segmentation or exact counting.

## 2. Choose the search scope

Use one of two approaches and record which was used.

### Evaluate an existing shortlist

Use this approach when specific models have already been proposed.

- Resolve each name to an exact Hugging Face repository ID, such as `organization/model-name`.
- Preserve explicitly requested checkpoints, including older generations.
- For an informal family name, check the publisher's current releases before choosing a checkpoint.
- Mention newer alternatives to an explicitly requested older model, but keep them separate from the requested evaluation.
- Add sibling sizes or other families only after agreeing to expand the scope.
- Record models that cannot be found or cannot be evaluated; do not silently omit them.

Apply the compatibility, license, and model-card checks below to the supplied list. If none qualify, state that outcome and agree on whether to broaden the search.

### Search the model hub

Use this approach when there is no shortlist.

1. Search Hugging Face using `pipeline_tag=image-text-to-text`.
2. Inspect results sorted by trending, downloads, and creation date. A bounded search of roughly 50-100 results per ordering is a reasonable starting point.
3. Cross-check the [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) for candidates at the required size and license.
4. Check the publishers' organization pages or release notes for current generations and available sizes.
5. Apply the initial license and compatibility checks before reviewing candidates in detail.

Hub search primarily matches repository names and metadata. It is not a semantic search of every model card, and a bounded search is not an exhaustive survey.

Use existing family lists as starting points, not as the complete candidate set. Newly released or less familiar models are eligible if their capabilities, licenses, and deployment path can be verified.

Prefer the newest verified release at the required size, provided it passes the deployment checks. A newer family release at a much larger size is not a replacement for a smaller checkpoint that fits the workload.

## 3. Apply the initial gates

### License eligibility

Classify each checkpoint for the intended deployment.

| Classification | Meaning | Action |
|---|---|---|
| Usable | Verified terms permit the deployment without special use restrictions | Continue evaluation; retain applicable notices and obligations |
| Usable with conditions | Terms permit the deployment subject to requirements such as registration, attribution, scale limits, or acceptable-use restrictions | Record the conditions and obtain acceptance before selecting |
| Not usable or unresolved | Terms conflict with the deployment, cannot be verified, or required access is unavailable | Exclude from the deployment shortlist and record why |

An unambiguous permissive tag such as Apache-2.0, MIT, or BSD can support initial triage. Review the actual terms during detailed evaluation.

For an `other`, missing, or unfamiliar license tag, follow the repository's license link immediately. Do not reject a candidate solely because its tag is ambiguous, and do not assume that an ambiguous tag permits commercial use.

Check licenses per checkpoint and size. Different releases in the same family may have different terms. Check weight licenses and relevant component licenses, not just the code license. Read any acceptable-use policy incorporated into the grant, particularly for surveillance-related applications.

### Runtime compatibility

Start with the current [OpenVINO supported-models page](https://docs.openvino.ai/nightly/documentation/compatibility-and-support/supported-models.html). Consult the linked runtime documentation and [OpenVINO GenAI supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) for pipeline-specific notes.

| Tier | Evidence | Required follow-up |
|---|---|---|
| Validated | The architecture is explicitly listed in the relevant support documentation | Verify the exact checkpoint, precision, pipeline, and target device |
| Plausible | The architecture resembles a supported model, for example a fine-tune of a supported base | Run an export and inference feasibility test |
| Unknown | No established support or sufficiently similar architecture was found | Treat as an engineering-risk option; establish feasibility before selection |

Here, **Validated means documented architecture support**, not measured application performance. Support for text generation does not establish support for the complete VLM pipeline. GPU support does not establish NPU support.

For an open search, reduce the list to roughly 8-10 candidates with usable licenses and Validated or Plausible compatibility. Retain an Unknown candidate only as a clearly identified stretch option when supported choices do not meet the requirements.

## 4. Review the shortlisted models

For each exact checkpoint, collect the following from the model card, configuration, publisher documentation, and runtime documentation.

| Area | Evidence to collect |
|---|---|
| Task relevance | Supported input/output types; image, multi-image, and video capabilities; grounding or OCR support where needed |
| Accuracy | Relevant benchmark results, evaluation conditions, source, and date |
| Memory and compute | Total parameters, supported precisions, vision encoder, image tiling, frame limits, and context limits |
| Deployment | Existing exports, required packages, version pins, and known export/runtime limitations |
| License | Actual terms, component restrictions, access requirements, and commercial-use conditions |
| Maintenance | Last update, active issues, downloads, and available deployment examples |

For unfamiliar models, verify every material claim against a current source. For all families, verify the current generation rather than relying on a remembered release name.

Downloads and likes indicate adoption, not accuracy. A community OpenVINO export is useful feasibility evidence, but it does not replace testing or license review.

### Use benchmarks that match the task

| Task | Example benchmark signals |
|---|---|
| OCR and documents | OCRBench, DocVQA |
| Charts | ChartQA |
| General visual reasoning | MMMU, MMBench |
| Video understanding | MVBench, Video-MME |

Use the following evidence order:

1. Third-party leaderboard results for the exact checkpoint and relevant task.
2. Self-reported model-card results, explicitly labeled.
3. Adoption and maintenance signals, only as weak usability evidence when benchmark results are unavailable.

If a leaderboard cannot be accessed or has no result for the checkpoint, record that limitation. Do not describe model-card results as leaderboard-verified.

Compare results only when the benchmark settings are compatible. Record differences such as subtitles, frame counts, resolution, prompting, or reasoning mode. Do not equate a benchmark score with application accuracy, and do not transfer an unquantized result to INT4 without validation.

### Handle missing results for a new generation

When a current checkpoint has no published relevant result, an older same-size checkpoint from the same family can provide a limited proxy.

- Use the nearest preceding generation with published results.
- Name both exact checkpoints.
- Label the number as a cross-generation proxy.
- Do not assign the older checkpoint's measured score to the newer one.
- Require application evaluation and, where needed, an export feasibility test.

For example: "No video result was found for the current 2B checkpoint. The previous generation's 2B result is included as a size-class proxy, not a measurement of the current model."

## 5. Check hardware fit

Evaluate the complete pipeline, not just model weights.

| Device | Main checks |
|---|---|
| Core Ultra NPU | Exact processor generation, driver/runtime support, supported operations and shapes, quantization requirements, and complete VLM pipeline coverage |
| Core Ultra iGPU | Shared-memory availability, image encoding cost, memory bandwidth, and contention with other workloads |
| Arc / Arc Pro discrete GPU | Exact SKU, dedicated VRAM, driver/runtime support, and concurrent-request capacity |

Allow for weights, vision inputs, activations, KV cache, runtime buffers, video decoding, and other application processes. Effective or active parameter counts do not necessarily represent total resident memory, particularly for embedding-heavy or mixture-of-experts models.

INT4 is a useful starting point for memory-constrained deployment; INT8 may be appropriate where quality loss is unacceptable and resources permit it. Measure the selected precision on the target device.

Do not convert vendor TOPS figures into a VLM latency estimate. Do not assume that a successful single-request test proves multi-camera capacity.

If no candidate fits, consider another available device tier: NPU, then iGPU, then discrete GPU. Document changes to hardware or performance requirements. Do not silently relax license requirements or accuracy targets.

## 6. Verify export and inference

Use a separate environment and the package versions required by the exact architecture. Record Python, OpenVINO, Optimum Intel, Transformers, quantization tooling, and driver versions.

A basic export probe is:

```bash
optimum-cli export openvino \
  --model <organization/model-name> \
  --weight-format int4 \
  <output-directory>
```

This is a template, not a universal device configuration. Apply any architecture-specific dependencies, version pins, and device-specific compression requirements from the current documentation.

After export, run the intended image or video input through the complete pipeline on the target device. Record the export result, runtime result, memory use, latency, and failure details.

For Plausible or Unknown candidates, an export failure is unresolved compatibility evidence. Do not turn it into a supported deployment claim.

### Gated repositories

The account owner must review and accept access terms on Hugging Face. Record any contact-information sharing, commercial registration, or use restrictions.

Use a locally configured Read-scope token when authentication is required. Do not put tokens in documentation, source code, chat, or logs. Verify access without printing credential values.

If access cannot be obtained, continue with another model rather than blocking the selection.

## 7. Rank the candidates

Use a comparison table that separates eligibility, evidence quality, and measured performance.

| Checkpoint / revision | Task fit | Accuracy and source | Compatibility tier | Device / precision | License status | Measured latency / throughput | Open issues |
|---|---|---|---|---|---|---|---|
| Candidate A | Supported capabilities | Metric, settings, source, date | Validated / Plausible / Unknown | Exact target | Usable / conditional / excluded | Measured result or not tested | Remaining checks |

Do not average away a hard constraint. A model with an incompatible license or a failed required capability does not qualify because it scores well elsewhere.

Select:

- **Primary:** the best-supported candidate for the stated workload and constraints.
- **Fallback:** an eligible alternative for a specific failure mode, such as export incompatibility, excessive latency, memory pressure, or access restrictions.

Explain the deciding trade-off. Separate published facts, estimates, and measurements. A model that has not passed hardware testing is a candidate for validation, not an approved production configuration.

## 8. Validate on application data

Run the exported model with representative inputs and the intended prompts, preprocessing, precision, and output limits.

At minimum, check:

- Task quality on labeled examples, including negative cases and difficult conditions.
- Precision, recall, and false-positive rates where the application detects events.
- JSON schema validity, required fields, and uncertainty handling for structured output.
- Latency distribution, including queueing, preprocessing, inference, and postprocessing.
- Sustained throughput and burst behavior at the required concurrency.
- Peak memory and behavior alongside other application workloads.
- Quality changes caused by quantization, reduced resolution, or frame sampling.
- Offline operation with all model, processor, tokenizer, and runtime dependencies available locally.

For video, test temporal events as sequences. A frame sampling rate that misses the relevant action cannot be corrected by choosing a larger model.

Keep published benchmark results separate from these application measurements.

## Selection record

The final record should contain:

1. Workload requirements and assumptions.
2. Search scope, date, and source links.
3. Exact checkpoints and revisions evaluated, including exclusions.
4. Comparison table with compatibility tiers and accuracy-source labels.
5. Primary model and fallback, with the deciding trade-offs.
6. Device placement, precision, export procedure, and dependency versions.
7. License conditions and access or custom-code approvals.
8. Hardware and application test results, or explicit untested items.
9. Remaining risks and the acceptance criteria for deployment.

Model releases, licenses, and runtime support change. Recheck them when the checkpoint, hardware, runtime, or deployment terms change.
