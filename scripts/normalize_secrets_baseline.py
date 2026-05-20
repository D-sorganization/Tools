#!/usr/bin/env python
"""Normalize .secrets.baseline paths to use forward slashes (POSIX-style).

Windows detect-secrets generates backslash paths; this script normalizes them
so the baseline is identical whether generated on Linux CI or Windows dev boxes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def normalize_baseline(input_path: Path, output_path: Path) -> None:
    """Read baseline JSON and normalize all path separators to forward slashes."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    new_results: dict = {}
    for k, v in data.get("results", {}).items():
        new_k = k.replace("\\", "/")
        new_v = []
        for entry in v:
            new_entry = dict(entry)
            new_entry["filename"] = new_entry["filename"].replace("\\", "/")
            new_v.append(new_entry)
        new_results[new_k] = new_v
    data["results"] = new_results

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/new_baseline_raw.json")
    dst = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("/tmp/new_baseline_normalized.json")
    )
    normalize_baseline(src, dst)
    print(f"Normalized {src} -> {dst}")
