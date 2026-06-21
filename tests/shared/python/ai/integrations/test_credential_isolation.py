"""Per-instance credential isolation tests for the integration clients.

These guard the Tools #3745 P2 cleanup that replaced module-level mutable
credential globals with per-consumer credentials/config objects. The key
property: constructing an independent credentials object never clobbers the
shared default instance (and vice versa), which is what defeats the old
cross-consumer global-state leak.
"""

from __future__ import annotations

from pathlib import Path

from ._bootstrap import bootstrap_integration_client_test

ROOT = Path(__file__).resolve().parents[5]
bootstrap_integration_client_test(ROOT)

import pytest  # noqa: E402

from src.shared.python.ai.integrations import affine as affine_mod  # noqa: E402
from src.shared.python.ai.integrations import linear as linear_mod  # noqa: E402
from src.shared.python.ai.integrations import notion as notion_mod  # noqa: E402
from src.shared.python.ai.integrations import obsidian as obsidian_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_defaults(monkeypatch: pytest.MonkeyPatch):
    """Clear all default credentials + env vars so tests do not leak state."""
    notion_mod.get_default_credentials().token = None
    linear_mod.get_default_credentials().token = None
    creds = affine_mod.get_default_credentials()
    creds.token = None
    creds.base_url = "https://app.affine.pro/graphql"
    obsidian_mod.get_default_config().vault_path = None
    for var in (
        "NOTION_API_KEY",
        "LINEAR_API_KEY",
        "AFFINE_API_KEY",
        "AFFINE_BASE_URL",
        "OBSIDIAN_VAULT_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
    notion_mod.get_default_credentials().token = None
    linear_mod.get_default_credentials().token = None
    creds = affine_mod.get_default_credentials()
    creds.token = None
    creds.base_url = "https://app.affine.pro/graphql"
    obsidian_mod.get_default_config().vault_path = None


@pytest.mark.unit
def test_notion_independent_instances_do_not_clobber() -> None:
    """A second NotionCredentials never overwrites the default instance."""
    notion_mod.set_notion_api_token("default-token")
    other = notion_mod.NotionCredentials(token="other-token")

    assert notion_mod.get_default_credentials().token == "default-token"
    assert other.token == "other-token"
    # Mutating one side leaves the other untouched.
    other.token = "changed"
    assert notion_mod.get_default_credentials().token == "default-token"
    assert other.resolve_token() == "changed"


@pytest.mark.unit
def test_linear_independent_instances_do_not_clobber() -> None:
    linear_mod.set_linear_api_token("default-token")
    other = linear_mod.LinearCredentials(token="other-token")

    assert linear_mod.get_default_credentials().token == "default-token"
    assert other.resolve_token() == "other-token"
    other.token = "changed"
    assert linear_mod.get_default_credentials().token == "default-token"


@pytest.mark.unit
def test_affine_independent_instances_do_not_clobber() -> None:
    affine_mod.set_affine_api_token("default-token")
    affine_mod.set_affine_base_url("https://self-hosted.example/graphql")
    other = affine_mod.AffineCredentials(token="other-token")

    assert affine_mod.get_default_credentials().token == "default-token"
    assert (
        affine_mod.get_default_credentials().resolve_base_url()
        == "https://self-hosted.example/graphql"
    )
    # The independent instance keeps the package default base URL.
    assert other.resolve_base_url() == "https://app.affine.pro/graphql"
    assert other.resolve_token() == "other-token"


@pytest.mark.unit
def test_obsidian_independent_configs_do_not_clobber(tmp_path: Path) -> None:
    vault_a = tmp_path / "a"
    vault_b = tmp_path / "b"
    vault_a.mkdir()
    vault_b.mkdir()

    obsidian_mod.set_obsidian_vault_path(vault_a)
    other = obsidian_mod.ObsidianConfig(vault_path=vault_b.resolve())

    assert obsidian_mod.get_default_config().vault_path == vault_a.resolve()
    assert other.resolve_vault_root() == vault_b.resolve()
    # Default config is unchanged after building an independent one.
    assert obsidian_mod.get_default_config().resolve_vault_root() == vault_a.resolve()


@pytest.mark.unit
def test_missing_credentials_still_raise() -> None:
    """Backward-compatible error path: unconfigured access raises ValueError."""
    with pytest.raises(ValueError, match="Notion API token not configured"):
        notion_mod.get_default_credentials().resolve_token()
    with pytest.raises(ValueError, match="Linear API token not configured"):
        linear_mod.get_default_credentials().resolve_token()
    with pytest.raises(ValueError, match="Affine API token not configured"):
        affine_mod.get_default_credentials().resolve_token()
    with pytest.raises(RuntimeError, match="not configured"):
        obsidian_mod.get_default_config().resolve_vault_root()
