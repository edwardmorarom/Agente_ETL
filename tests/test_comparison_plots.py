from __future__ import annotations

import pandas as pd

from core.comparison_plots import build_comparison_plots


def test_build_comparison_plots_returns_embeddable_html_for_each_plot() -> None:
    df_before = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    df_after = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    plots = build_comparison_plots(df_before, df_after, ["x"])

    assert set(plots) == {"x"}
    assert set(plots["x"]) == {"boxplot", "histograma", "qqplot"}
    for html_fragment in plots["x"].values():
        assert "<div" in html_fragment
        assert ".png" not in html_fragment


def test_build_comparison_plots_returns_empty_dict_without_missing_vars() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    assert build_comparison_plots(df, df, []) == {}
