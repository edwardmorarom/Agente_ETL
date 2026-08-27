from __future__ import annotations

import shutil

import pandas as pd
import pytest

from imputers.mice_imputer import MiceImputer
from imputers.regresion_imputer import RegresionImputer


pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript no esta disponible en el sistema.",
)


def test_mice_imputer_runs_real_r_script() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, None, 3.5, 4.2, 5.8, 7.1],
            "y": [2.2, 3.9, None, 8.4, 9.7, 13.1],
            "z": [5.0, 1.4, 6.2, None, 3.3, 8.9],
        }
    )

    imputer = MiceImputer(m=2, maxit=1, seed=123)
    result = imputer.fit_transform(df)

    assert result.isna().sum().sum() == 0
    assert imputer.last_report is not None
    assert "case" in imputer.last_report
    assert "beta_distribution_fit" not in imputer.last_report


def test_mice_imputer_runs_real_r_script_with_beta_vars() -> None:
    df = pd.DataFrame(
        {
            "pct": [12.0, 25.0, None, 44.0, 51.0, 67.0, None, 82.0, 91.0, 36.0],
            "helper": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )

    imputer = MiceImputer(beta_vars=["pct"], m=2, maxit=1, seed=123)
    result = imputer.fit_transform(df)

    assert result.isna().sum().sum() == 0
    assert imputer.last_report is not None
    assert "beta_distribution_fit" in imputer.last_report
    beta_fit = imputer.last_report["beta_distribution_fit"]["pct"]
    assert beta_fit["scale_used"] == "percentage_0_100"
    assert beta_fit["n_imputations"] == 2
    assert "shape1_pooled" in beta_fit
    assert "shape2_pooled" in beta_fit
    assert "severity_joint" in beta_fit


def test_regresion_imputer_runs_real_r_script() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, None, 10.0, 12.0],
            "z": [1.0, 1.5, 2.0, 2.5, None, 3.5],
        }
    )

    imputer = RegresionImputer()
    result = imputer.fit_transform(df)

    assert result.isna().sum().sum() == 0
