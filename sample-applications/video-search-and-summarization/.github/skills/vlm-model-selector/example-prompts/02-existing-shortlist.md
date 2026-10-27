<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

I have an existing shortlist of one: Qwen/Qwen2.5-VL-7B-Instruct. Evaluate
that exact checkpoint for answering questions about scanned invoices. Input
is one image per request, and the output is English JSON containing the answer
and supporting text. It must run commercially and offline on a 32 GB Core
Ultra iGPU. INT4 is acceptable; the target is 5 seconds per request.

Check the current model card, actual license terms, relevant document
benchmarks, and OpenVINO compatibility. If a newer generation exists, mention
it separately without replacing my requested checkpoint or adding it to the
evaluated shortlist. Provide a proposed export command and remaining checks,
but do not download weights, install packages, or run the export.
