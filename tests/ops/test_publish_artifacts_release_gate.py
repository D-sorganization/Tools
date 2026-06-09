from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_npm_publish_uses_protected_environment() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "publish-artifacts.yml").read_text(
            encoding="utf-8"
        )
    )

    npm_job = workflow["jobs"]["publish-npm"]

    assert npm_job["environment"]["name"] == "npm"
    assert "NPM_TOKEN" in str(npm_job["steps"])
