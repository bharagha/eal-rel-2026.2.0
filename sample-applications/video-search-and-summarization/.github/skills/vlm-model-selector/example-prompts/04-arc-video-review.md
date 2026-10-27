<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

Search broadly for a VLM to summarize recorded loading-dock video clips and
answer questions about vehicle arrivals and unloading activity. Input is
30-second clips with sampled frames; output is an English paragraph and JSON
event timestamps. The target hardware is an Intel Arc Pro B60 with 24 GB
VRAM and 64 GB host RAM. Verify the device configuration and runtime support.

This is commercial and fully local at runtime. INT4 or INT8 is acceptable.
There is no strict real-time deadline; accuracy is the main priority within
the hardware budget. Compare relevant video benchmarks, current licenses,
image/frame token costs, and export limitations. Recommend a primary and
fallback, noting any gated-access or custom-code requirements before taking
action. Do not accept terms, request token values, download models, or execute
repository code.
