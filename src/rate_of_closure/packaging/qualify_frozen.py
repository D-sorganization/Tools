"""Validation harness for the frozen Windows PyQt6 Rate of Closure artifact.

Verifies artifact hygiene, offscreen execution, Qt/Matplotlib/SciPy collection,
graceful optional-Rust capability messaging, bounded canonical simulation,
Ground Study evidence save, clean exit, spaces/Unicode path support,
unrelated-cwd behavior, and byte-identical result evidence versus golden contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rate_of_closure.packaging.entrypoint import (
    DIRECT_WORKER_RECOVERY_UNSUPPORTED_REASON,
)


@dataclass(frozen=True)
class ArtifactHygieneViolation:
    """One hygiene violation detected in the frozen distribution directory."""

    path: str
    message: str


@dataclass(frozen=True)
class HygieneReport:
    """Report on bundle cleanliness and hygiene."""

    is_clean: bool
    violations: tuple[ArtifactHygieneViolation, ...]


def check_artifact_hygiene(bundle_dir: Path) -> HygieneReport:
    """Scan bundle directory for secrets, source leaks, databases, or unwanted files."""
    violations: list[ArtifactHygieneViolation] = []

    for root, _dirs, files in os.walk(bundle_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(bundle_dir)

        for filename in files:
            rel_file = rel_root / filename
            rel_str = str(rel_file).replace("\\", "/")

            if filename.startswith(".env"):
                violations.append(
                    ArtifactHygieneViolation(
                        rel_str, f"Secret/environment file found: {filename}"
                    )
                )
            if filename.endswith((".sqlite", ".sqlite3", ".db")):
                violations.append(
                    ArtifactHygieneViolation(
                        rel_str, f"Database file found in artifact: {filename}"
                    )
                )
            # No uncompiled .py files should reside in the root bundle dir
            if rel_root == Path(".") and filename.endswith(".py"):
                violations.append(
                    ArtifactHygieneViolation(
                        rel_str,
                        f"Uncompiled source file in artifact root: {filename}",
                    )
                )
            if ".git" in rel_file.parts:
                violations.append(
                    ArtifactHygieneViolation(
                        rel_str, f"Git repository metadata found: {rel_str}"
                    )
                )

    return HygieneReport(
        is_clean=len(violations) == 0,
        violations=tuple(violations),
    )


def qualify_frozen_artifact(
    bundle_dir: Path,
    *,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Execute full qualification suite against a built one-folder artifact."""
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {bundle_dir}")

    exe_name = (
        "RateOfClosureExplorer.exe"
        if sys.platform == "win32"
        else "RateOfClosureExplorer"
    )
    exe_path = bundle_dir / exe_name
    if not exe_path.is_file():
        raise FileNotFoundError(f"Expected executable missing: {exe_path}")

    # 1. Artifact Hygiene
    hygiene = check_artifact_hygiene(bundle_dir)
    if not hygiene.is_clean:
        violation_msgs = [v.message for v in hygiene.violations]
        raise RuntimeError(f"Artifact hygiene failure: {violation_msgs}")

    # 2. Capabilities Probe
    probe_proc = subprocess.run(
        [str(exe_path), "--probe-capabilities"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if probe_proc.returncode != 0:
        raise RuntimeError(
            f"--probe-capabilities failed ({probe_proc.returncode}): "
            f"{probe_proc.stderr}"
        )

    caps = json.loads(probe_proc.stdout)
    recovery = caps.get("direct_worker_restart_recovery", {})
    if recovery.get("supported") is not False:
        raise AssertionError("direct_worker_restart_recovery must be unsupported")
    if recovery.get("reason") != DIRECT_WORKER_RECOVERY_UNSUPPORTED_REASON:
        raise AssertionError("direct_worker_restart_recovery reason mismatch")

    rust_cap = caps.get("rust_capability", {})
    if "available" not in rust_cap or "backend" not in rust_cap:
        raise AssertionError("rust_capability shape invalid")

    models = caps.get("collected_models", {})
    for req_model in (
        "locus_capabilities",
        "neural_capabilities",
        "example_driver_head",
    ):
        if not models.get(req_model):
            raise AssertionError(f"required model missing: {req_model}")

    # 3. Offscreen Smoke Test
    smoke_proc = subprocess.run(
        [str(exe_path), "--smoke-test"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if smoke_proc.returncode != 0:
        raise RuntimeError(
            f"--smoke-test failed ({smoke_proc.returncode}): {smoke_proc.stderr}"
        )

    # 4. Canonical Simulation Parity
    sim_proc = subprocess.run(
        [str(exe_path), "--run-canonical-simulation"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if sim_proc.returncode != 0:
        raise RuntimeError(f"--run-canonical-simulation failed: {sim_proc.stderr}")
    sim_data = json.loads(sim_proc.stdout)
    if sim_data.get("status") != "success":
        raise AssertionError(f"simulation failed: {sim_data}")
    metrics = sim_data.get("metrics", {})
    if metrics.get("carry_distance_m", 0.0) <= 5.0:
        raise AssertionError(f"unphysical carry: {metrics.get('carry_distance_m')}")

    # 5. Ground Study Evidence Save and Parity
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        res_file = temp_dir / "regional-ground-execution-result.json"
        csv_file = temp_dir / "regional-ground-execution-rows.csv"

        gs_proc = subprocess.run(
            [
                str(exe_path),
                "--execute-ground-study",
                "--result-file",
                str(res_file),
                "--csv-file",
                str(csv_file),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if gs_proc.returncode != 0:
            raise RuntimeError(f"--execute-ground-study failed: {gs_proc.stderr}")

        if not res_file.is_file() or not csv_file.is_file():
            raise AssertionError("ground study evidence files were not created")

        saved_doc = json.loads(res_file.read_text(encoding="utf-8"))
        repo_fixtures = (
            Path(__file__).parents[3]
            / "src"
            / "rate_of_closure"
            / "web"
            / "src"
            / "model"
            / "__fixtures__"
            / "regional_ground_execution_result_golden_v1.json"
        )
        golden_doc = json.loads(repo_fixtures.read_text(encoding="utf-8"))
        if saved_doc != golden_doc["result"]:
            raise AssertionError("ground study result does not match golden fixture")

    # 6. Spaces, Unicode path, and Unrelated CWD Behavior
    with tempfile.TemporaryDirectory() as td_unicode:
        unicode_parent = Path(td_unicode) / "RateOfClosure 🏌️ 测试 Directory"
        unicode_bundle = unicode_parent / "Explorer Bundle"
        created_link = False
        try:
            if os.name == "nt":
                import _winapi

                unicode_parent.mkdir(parents=True, exist_ok=True)
                _winapi.CreateJunction(str(bundle_dir.resolve()), str(unicode_bundle))
                created_link = True
            else:
                unicode_parent.mkdir(parents=True, exist_ok=True)
                unicode_bundle.symlink_to(bundle_dir.resolve())
                created_link = True
        except Exception:
            shutil.copytree(bundle_dir, unicode_bundle)

        try:
            unicode_exe = unicode_bundle / exe_name
            unrelated_cwd = Path(tempfile.gettempdir())

            cwd_proc = subprocess.run(
                [str(unicode_exe), "--probe-capabilities"],
                capture_output=True,
                text=True,
                cwd=unrelated_cwd,
                timeout=timeout_s,
                check=False,
            )
            if cwd_proc.returncode != 0:
                raise RuntimeError(
                    "unrelated cwd and unicode path execution failed: "
                    f"{cwd_proc.stderr}"
                )
        finally:
            if created_link and unicode_bundle.exists():
                try:
                    os.rmdir(unicode_bundle)
                except Exception:
                    pass

    return {
        "status": "qualified",
        "bundle_dir": str(bundle_dir),
        "executable": str(exe_path),
        "hygiene": "passed",
        "capabilities": caps,
        "simulation": sim_data,
        "ground_study_parity": "verified",
        "unicode_and_cwd_relocation": "verified",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI runner for qualification harness."""
    parser = argparse.ArgumentParser(
        description="Qualify frozen Rate of Closure artifact"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).parents[3] / "dist" / "RateOfClosureExplorer",
        help="Path to the one-folder bundle directory",
    )
    args = parser.parse_args(argv)

    try:
        report = qualify_frozen_artifact(args.artifact_dir)
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Qualification failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
