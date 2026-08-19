# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash
TOOL_ROOT := $(shell cd "$(dir $(lastword $(MAKEFILE_LIST)))" && pwd)

.PHONY: help bench lint test clean

help:
	@echo "Targets:"
	@echo "  bench   - run the interactive benchmark (./run_benchmark.sh)"
	@echo "  lint    - shellcheck + python lint on all scripts"
	@echo "  test    - run unit tests (tests/)"
	@echo "  clean   - remove results/, model-cache/, and cached tool downloads"

bench:
	@"$(TOOL_ROOT)/run_benchmark.sh"

lint:
	@command -v shellcheck >/dev/null 2>&1 && shellcheck "$(TOOL_ROOT)"/*.sh "$(TOOL_ROOT)"/scripts/*.sh "$(TOOL_ROOT)"/scripts/lib/*.sh || echo "shellcheck not installed, skipping shell lint"
	@command -v ruff >/dev/null 2>&1 && ruff check "$(TOOL_ROOT)/scripts" "$(TOOL_ROOT)/tests" || python3 -m py_compile "$(TOOL_ROOT)"/scripts/*.py "$(TOOL_ROOT)"/tests/*.py

test:
	@python3 -m pytest "$(TOOL_ROOT)/tests" -v

clean:
	@rm -rf "$(TOOL_ROOT)/results" "$(TOOL_ROOT)/model-cache" "$(TOOL_ROOT)/.export-tool" "$(TOOL_ROOT)/.bench-tool"
