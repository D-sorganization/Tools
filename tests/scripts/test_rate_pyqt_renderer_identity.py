"""A screenshot identity cannot silently represent several font renderers."""

import json
from pathlib import Path

import pytest

from scripts import check_rate_pyqt_environment as environment_check

ROOT = Path(__file__).resolve().parents[2]
HOSTED_STACK = {
    "libfontconfig1": "2.15.0-1.1ubuntu2",
    "libfreetype6": "2.13.2+dfsg-1ubuntu0.1",
    "matplotlib_freetype": "2.14.3",
}


def test_repository_selects_one_exact_hosted_font_stack() -> None:
    """Both capture paths must use the same qualified renderer (#4844)."""
    path = environment_check.DEFAULT_FONT_STACK_EXPECTATIONS
    assert json.loads(path.read_text(encoding="utf-8")) == HOSTED_STACK


@pytest.mark.parametrize(
    "invalid",
    [[], {}, {"libfreetype6": ["a", "b"]}, {"libfreetype6": ""}, {"invented": "1"}],
)
def test_ambiguous_font_authority_is_rejected_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    path = tmp_path / "fonts.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    def unexpected_probe() -> dict[str, str]:
        pytest.fail("invalid font authority must fail before reading the host")

    monkeypatch.setattr(environment_check, "probe_font_stack", unexpected_probe)
    with pytest.raises(ValueError, match="font stack"):
        environment_check.verify_font_stack(path)


def test_known_other_host_is_not_the_approved_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        environment_check,
        "probe_font_stack",
        lambda: {
            **HOSTED_STACK,
            "libfontconfig1": "2.17.1-3ubuntu1",
            "libfreetype6": "2.14.2+dfsg-1ubuntu0.1",
        },
    )
    with pytest.raises(RuntimeError, match="libfontconfig1.*libfreetype6"):
        environment_check.verify_font_stack(
            environment_check.DEFAULT_FONT_STACK_EXPECTATIONS
        )


def test_qt_binary_and_binding_versions_are_verified() -> None:
    expected = environment_check.read_expected_versions(
        ROOT / "requirements-rate-pyqt.txt"
    )
    assert expected["pyqt6-qt6"] == "6.11.2"
    assert expected["pyqt6-sip"] == "13.12.0"
