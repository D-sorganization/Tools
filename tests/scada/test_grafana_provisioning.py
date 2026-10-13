"""Contract and lint tests for Grafana provisioning-as-code and dashboards.

H6 / #5545:
1. Datasource YAML parses, uses env-var interpolation for credentials,
   and configures PostgreSQL / TimescaleDB.
2. Dashboard provider YAML parses and configures file-based provisioning.
3. Plant-overview dashboard JSON parses, has a stable UID, and panel queries reference
   the H3 continuous aggregate (tag_sample_1m).
4. No Grafana binary or container image is vendored.
5. README documents licensing (AGPLv3 internal-only #4951) and safety invariants.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

# Repository root: tests/scada/ -> root
ROOT = Path(__file__).resolve().parents[2]
GRAFANA_DIR = ROOT / "deploy" / "historian" / "grafana"
DATASOURCES_DIR = GRAFANA_DIR / "provisioning" / "datasources"
DASHBOARDS_PROV_DIR = GRAFANA_DIR / "provisioning" / "dashboards"
DASHBOARDS_DIR = GRAFANA_DIR / "dashboards"

_LITERAL_SECRET_RE = re.compile(
    r"(?:password\s*:\s*['\"][^\$\{\n'\"]+['\"]"
    r"|password\s*:\s*(?!\$\{)[a-zA-Z0-9_\-]+"
    r"|postgres(?:ql)?://[^:]+:[^@]+@)",
    re.IGNORECASE,
)

_ENV_INTERPOLATION_RE = re.compile(r"\$\{[A-Za-z0-9_]+(?::-?[^}]*)?\}")

_DISALLOWED_EXTENSIONS = {
    ".exe",
    ".bin",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".iso",
    ".img",
    ".so",
    ".dll",
    ".dylib",
}


def test_grafana_directories_exist() -> None:
    """deploy/historian/grafana/ and required subdirectories must exist."""
    assert GRAFANA_DIR.is_dir(), f"Missing grafana root directory: {GRAFANA_DIR}"
    assert DATASOURCES_DIR.is_dir(), f"Missing datasources dir: {DATASOURCES_DIR}"
    assert DASHBOARDS_PROV_DIR.is_dir(), (
        f"Missing dashboards prov dir: {DASHBOARDS_PROV_DIR}"
    )
    assert DASHBOARDS_DIR.is_dir(), f"Missing dashboards dir: {DASHBOARDS_DIR}"


def test_datasource_yaml_parses_and_has_valid_timescale_config() -> None:
    """timescale.yaml must parse and configure TimescaleDB over PostgreSQL."""
    ds_file = DATASOURCES_DIR / "timescale.yaml"
    assert ds_file.is_file(), f"Missing datasource file: {ds_file}"

    with open(ds_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), f"Expected dict in {ds_file}, got {type(data)}"
    assert data.get("apiVersion") == 1, "Expected apiVersion: 1"

    datasources = data.get("datasources", [])
    assert isinstance(datasources, list) and len(datasources) >= 1, (
        "Expected at least one datasource in timescale.yaml"
    )

    timescale_ds = next(
        (ds for ds in datasources if ds.get("type") == "postgres"), None
    )
    assert timescale_ds is not None, "Expected a datasource of type 'postgres'"

    json_data = timescale_ds.get("jsonData", {})
    assert json_data.get("timescaledb") is True, (
        "Expected jsonData.timescaledb to be True for TimescaleDB datasource"
    )


def test_datasource_credentials_use_env_interpolation_only() -> None:
    """Datasource credentials must use env-var interpolation only."""
    ds_file = DATASOURCES_DIR / "timescale.yaml"
    assert ds_file.is_file(), f"Missing datasource file: {ds_file}"

    content = ds_file.read_text(encoding="utf-8")

    # Reject literal password assignments
    matches = _LITERAL_SECRET_RE.findall(content)
    assert not matches, f"Found potential literal secrets in {ds_file}: {matches}"

    # Confirm environment variable interpolation is present for password
    with open(ds_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    ds = data["datasources"][0]
    secure_json = ds.get("secureJsonData", {})
    password_val = secure_json.get("password", "")

    assert _ENV_INTERPOLATION_RE.search(str(password_val)), (
        f"Password must use env-var interpolation, got: {password_val}"
    )


def test_dashboards_provider_yaml_parses_and_has_valid_structure() -> None:
    """dashboards.yaml must parse and declare a file provider pointing to dashboards."""
    dashboards_prov_file = DASHBOARDS_PROV_DIR / "dashboards.yaml"
    assert dashboards_prov_file.is_file(), (
        f"Missing dashboards provider file: {dashboards_prov_file}"
    )

    with open(dashboards_prov_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), f"Expected dict in {dashboards_prov_file}"
    assert data.get("apiVersion") == 1, "Expected apiVersion: 1"

    providers = data.get("providers", [])
    assert isinstance(providers, list) and len(providers) >= 1, (
        "Expected at least one provider in dashboards.yaml"
    )

    file_provider = next((p for p in providers if p.get("type") == "file"), None)
    assert file_provider is not None, "Expected at least one provider of type 'file'"
    assert "options" in file_provider and "path" in file_provider["options"], (
        "File provider must specify options.path"
    )


def test_plant_overview_dashboard_parses_and_has_stable_uid() -> None:
    """plant-overview.json must parse and specify a stable UID and title."""
    dashboard_file = DASHBOARDS_DIR / "plant-overview.json"
    assert dashboard_file.is_file(), f"Missing dashboard file: {dashboard_file}"

    with open(dashboard_file, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict), f"Expected dashboard JSON object, got {type(data)}"
    uid = data.get("uid")
    assert uid and isinstance(uid, str), (
        f"Dashboard must have a non-empty string uid, got {uid}"
    )
    assert len(uid) >= 3, f"Dashboard uid too short: {uid}"

    title = data.get("title")
    assert title and isinstance(title, str), f"Dashboard must have a title, got {title}"


def test_dashboard_panels_reference_continuous_aggregates() -> None:
    """Dashboard panels must query the H3 continuous aggregate (tag_sample_1m)."""
    dashboard_file = DASHBOARDS_DIR / "plant-overview.json"
    assert dashboard_file.is_file(), f"Missing dashboard file: {dashboard_file}"

    with open(dashboard_file, encoding="utf-8") as f:
        data = json.load(f)

    panels: list[dict[str, Any]] = data.get("panels", [])
    assert len(panels) >= 1, "Dashboard must have at least one panel"

    # Search queries / rawSql across all panels
    all_raw_queries: list[str] = []
    for panel in panels:
        targets = panel.get("targets", [])
        for target in targets:
            raw_sql = target.get("rawSql") or target.get("query") or ""
            if raw_sql:
                all_raw_queries.append(raw_sql)

    assert len(all_raw_queries) >= 1, (
        "Expected at least one panel target with a SQL query"
    )

    # Confirm tag_sample_1m is referenced
    cagg_found = any("tag_sample_1m" in q for q in all_raw_queries)
    assert cagg_found, (
        f"Expected at least one query in {dashboard_file} to reference "
        f"continuous aggregate 'tag_sample_1m'. Queries found: {all_raw_queries}"
    )


def test_no_binaries_or_container_images_vendored() -> None:
    """deploy/historian/grafana/ must not vendor binaries or large container images."""
    assert GRAFANA_DIR.is_dir(), f"Missing grafana root directory: {GRAFANA_DIR}"

    for path in GRAFANA_DIR.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in _DISALLOWED_EXTENSIONS, (
                f"Disallowed binary/archive extension in grafana deployment: {path}"
            )
            size_bytes = path.stat().st_size
            assert size_bytes < 500 * 1024, (
                f"File exceeds maximum allowed size (500KB): "
                f"{path} ({size_bytes} bytes)"
            )


def test_grafana_readme_documents_licensing_and_safety() -> None:
    """deploy/historian/grafana/README.md must document licensing and safety."""
    readme_file = GRAFANA_DIR / "README.md"
    assert readme_file.is_file(), f"Missing README: {readme_file}"

    content = readme_file.read_text(encoding="utf-8").lower()
    assert "agpl" in content or "4951" in content or "internal-only" in content, (
        "README must document AGPLv3 / internal-only licensing ruling (#4951)"
    )
    assert "safety" in content or "non-authoritative" in content, (
        "README must document non-authoritative / safety-critical invariants"
    )
