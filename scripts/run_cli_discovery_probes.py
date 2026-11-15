#!/usr/bin/env python3
"""Run Tier-2 subscription CLI discovery probes for all default providers (#5727)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _build_report() -> dict[str, Any]:
    from shared.python.ai.adapters.cli_qualification import (
        DEFAULT_CLI_DISCOVERY_SPECS,
        detect_forbidden_api_key_env,
        probe_cli_provider_discovery,
    )

    forbidden = detect_forbidden_api_key_env()
    if forbidden:
        logger.error(
            "Refusing discovery run: forbidden API-key env set (%s)",
            ", ".join(forbidden),
        )
        return {
            "schema": "tools/cli-discovery-report/v1",
            "error": "forbidden_api_key_env",
            "forbidden_env": list(forbidden),
            "probes": [],
        }

    probes = [
        asdict(
            probe_cli_provider_discovery(
                spec.provider,
                spec.binary_name,
                spec.required_help_flags,
            )
        )
        for spec in DEFAULT_CLI_DISCOVERY_SPECS
    ]
    return {
        "schema": "tools/cli-discovery-report/v1",
        "error": None,
        "forbidden_env": [],
        "probes": probes,
        "all_probe_ok": all(item["probe_ok"] for item in probes),
    }


def main(argv: list[str] | None = None) -> int:
    """Run discovery probes and write a JSON report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _ensure_src_on_path()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the JSON discovery report",
    )
    args = parser.parse_args(argv)

    report = _build_report()
    if report.get("error") == "forbidden_api_key_env":
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Wrote discovery report to %s (all_probe_ok=%s)",
        args.output,
        report["all_probe_ok"],
    )
    return 0 if report["all_probe_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
