from __future__ import annotations

from scripts.generate_pdf_report import _assumption_rows


def test_distribution_comparison_uses_ad_statistic_and_default_label() -> None:
    assumptions = {
        "distribution_comparison": {
            "x": {
                "ad_statistic": 1.23,
                "p_value": 0.04,
                "meets_assumption": False,
            }
        }
    }

    rows, failing_indexes = _assumption_rows(assumptions)

    assert rows[1] == [
        "x",
        "distribution_comparison",
        "Anderson-Darling (k-muestras)",
        "stat=1.23; p=0.04",
        "False",
    ]
    assert failing_indexes == [1]
