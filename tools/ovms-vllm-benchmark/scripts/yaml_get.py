#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Small helper to read a dotted-path value out of config/models.yaml.

Used by the bash orchestration scripts (which have no native YAML support)
to fetch model/engine metadata in a robust, testable way.

Usage:
    python3 yaml_get.py <config_path> <dotted.key.path> [--default DEFAULT]

Examples:
    python3 yaml_get.py config/models.yaml models.llm.hf_repo
    python3 yaml_get.py config/models.yaml engines.ovms.image
    python3 yaml_get.py config/models.yaml models.llm.ovms.export_extra_args
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def get_path(data: dict, dotted_path: str):
    node = data
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_path)
        node = node[part]
    return node


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("dotted_path")
    parser.add_argument("--default", default=None)
    args = parser.parse_args()

    with args.config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    try:
        value = get_path(data, args.dotted_path)
    except KeyError:
        if args.default is not None:
            print(args.default)
            return 0
        print(f"error: key not found: {args.dotted_path}", file=sys.stderr)
        return 1

    if isinstance(value, (dict, list)):
        print(json.dumps(value))
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
