<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

Search broadly for a compact VLM that describes visible surface defects on
manufactured parts in one short English sentence. Each request contains a
single conveyor-camera image. The hardware is a 32 GB Lunar Lake machine;
NPU-only inference is preferred for power reasons. Commercial offline use
and INT4 are required. The initial target is one result every 2 seconds.

Verify complete VLM NPU support rather than assuming that GPU or text-only
support is sufficient. Compare current checkpoints and distinguish general
visual benchmark scores from defect-description accuracy. If no NPU candidate
qualifies, explain whether the available iGPU is a viable alternative without
silently changing the target. Include the offline asset requirements and
validation plan; do not install or export anything.
