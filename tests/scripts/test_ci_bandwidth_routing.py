from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/ci-standard.yml")


def _workflow_jobs() -> dict[str, object]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    assert isinstance(jobs, dict)
    return jobs


def test_quality_gate_uses_hosted_compute_without_local_dispatch() -> None:
    jobs = _workflow_jobs()
    quality_gate = jobs["quality-gate"]

    assert isinstance(quality_gate, dict)
    assert quality_gate["runs-on"] == "ubuntu-24.04"
    assert "needs" not in quality_gate


def test_quality_gate_preserves_pip_download_cache() -> None:
    jobs = _workflow_jobs()
    quality_gate = jobs["quality-gate"]
    assert isinstance(quality_gate, dict)
    serialized = yaml.safe_dump(quality_gate)

    assert "PIP_NO_CACHE_DIR" not in serialized
    assert "pip cache purge" not in serialized
    assert "${{ runner.temp }}/pip-quality-gate" not in serialized
