from pathlib import Path

from scripts.check_tracked_text_normalization import non_lf_index_paths

ROOT = Path(__file__).resolve().parents[2]


def test_rejects_crlf_and_mixed_index_blobs_governed_as_lf() -> None:
    records = (
        "i/crlf  w/lf    attr/text eol=lf\tassessments/report.json",
        "i/mixed w/lf    attr/text eol=lf\tdocs/evidence.json",
        "i/lf    w/lf    attr/text eol=lf\tsrc/module.py",
    )

    assert non_lf_index_paths(records) == (
        "assessments/report.json",
        "docs/evidence.json",
    )


def test_allows_lf_empty_and_unmanaged_index_blobs() -> None:
    records = (
        "i/lf    w/lf    attr/text eol=lf\tsrc/module.py",
        "i/none  w/none  attr/text eol=lf\tempty.json",
        "i/crlf  w/crlf  attr/text=auto\tlegacy.txt",
        "i/-text w/-text attr/-text\tasset.bin",
    )

    assert non_lf_index_paths(records) == ()


def test_preserves_paths_containing_spaces() -> None:
    records = ("i/crlf  w/lf    attr/text eol=lf\tdocs/reviewer evidence.json",)

    assert non_lf_index_paths(records) == ("docs/reviewer evidence.json",)


def test_normalization_gate_is_wired_into_local_and_protected_checks() -> None:
    pre_commit = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    standard_ci = (ROOT / ".github/workflows/ci-standard.yml").read_text(
        encoding="utf-8"
    )
    distribution_ci = (
        ROOT / ".github/workflows/rate-of-closure-web-distribution.yml"
    ).read_text(encoding="utf-8")
    command = "python scripts/check_tracked_text_normalization.py"

    assert command in pre_commit
    assert command in standard_ci
    assert command in distribution_ci
