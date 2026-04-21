"""Regression tests for the decomposed ANOVA facade."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from data_processor.core.anova import ANOVAAnalyzer


def test_anova_facade_uses_extracted_modules(repo_root: Path) -> None:
    anova_path = (
        repo_root
        / "src"
        / "data_processing"
        / "data_processor"
        / "python"
        / "data_processor"
        / "core"
        / "anova.py"
    )
    content = anova_path.read_text(encoding="utf-8")

    assert "anova_models" in content
    assert "anova_one_way" in content
    assert "anova_two_way" in content
    assert "anova_repeated" in content


def test_one_way_anova_accepts_group_dictionary() -> None:
    analyzer = ANOVAAnalyzer()
    groups = {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([4.0, 5.0, 6.0]),
        "C": np.array([7.0, 8.0, 9.0]),
    }

    result = analyzer.one_way_anova(groups)

    assert result.df_between == 2
    assert len(result.group_means) == 3


def test_two_way_anova_still_returns_summary_table() -> None:
    analyzer = ANOVAAnalyzer()
    df = pd.DataFrame(
        {
            "value": [10.0, 12.0, 13.0, 15.0, 16.0, 18.0],
            "factor_a": ["low", "low", "high", "high", "low", "high"],
            "factor_b": ["x", "y", "x", "y", "x", "y"],
        }
    )

    result = analyzer.two_way_anova(df, "value", "factor_a", "factor_b")

    assert result.anova_table.source[0] == "factor_a"
    assert result.df_error >= 0
