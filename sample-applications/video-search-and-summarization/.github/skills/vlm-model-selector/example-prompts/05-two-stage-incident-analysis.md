<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

Search broadly for two VLMs for a local incident-review cascade on a 64 GB
Panther Lake machine. There are 8 live camera streams sampled at 1 frame per
second each. A small model screens frame sequences for possible item
concealment, cable damage, fire, or water accumulation and triggers a larger
model to produce English JSON with observable evidence and uncertainty.

Up to 3 cameras can trigger together. The deadline is under 10 seconds from
the small-model trigger to completed larger-model JSON, including queueing.
The sustained trigger rate is not yet known. Commercial offline use and INT4
are required. Do not use the system as the only fire alarm or treat a possible
concealment event as proof of theft.

Recommend a primary pair and an alternative. Distinguish published benchmarks
from application accuracy, verify device-specific compatibility, and state
the input budget and load tests needed before claiming the deadline. Identify
any missing workload parameters. Do not install dependencies, download
weights, or execute exports.
