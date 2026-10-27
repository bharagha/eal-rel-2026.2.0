<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

Search broadly for a VLM to review customer-employee interactions and possible
item concealment in a retail store. I have a 32 GB Panther Lake system and want
to use its iGPU. Input is sampled live video from 8 cameras, initially 1 frame
per second per camera. Return English JSON events with observable evidence
and an uncertain state; do not infer theft or intent from appearance alone.

This is a commercial, fully local deployment. INT4/INT8 is acceptable. The
target is a completed alert within 30 seconds of the relevant observation.
Compare current checkpoints, label accuracy sources and OpenVINO support,
and recommend a primary and fallback. Separate memory fit from measured
throughput and state what testing is required on my hardware. Do not download
weights or execute exports.
