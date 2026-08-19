"""Clean-install verification entry point for the packaged web distribution."""

from __future__ import annotations

import argparse
import sys

from .package_assets import resolve_packaged_web_assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True)
    arguments = parser.parse_args()
    bundle = resolve_packaged_web_assets()
    if bundle.release_revision != arguments.expected_revision:
        raise ValueError("installed web distribution revision does not match release")
    sys.stdout.write(
        f"verified {len(bundle.assets)} immutable static-inspection web assets\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
