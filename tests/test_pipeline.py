from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import core.pipeline as pipeline_module
from core.pipeline import ImputationPipeline


class FakeMiceImputer:
    created: list["FakeMiceImputer"] = []
    next_report: dict[str, Any] = {
        "severity": {"lambda": 0.1},
    }

    def __init__(
        self,
        vars: list[str] | None = None,
        m: int = 5,
        beta_vars: list[str] | None = None,
    ) -> None:
        self.vars = vars
        self.m = m
        self.beta_vars = beta_vars
        self.last_report = self.next_report
        self.fit_called = False
        self.created.append(self)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit_called = True
        return df.fillna(0)


class FakeRegresionImputer:
    created: list["FakeRegresionImputer"] = []

    def __init__(self, method: str = "stochastic_regression") -> None:
        self.method = method
        self.fit_called = False
        self.created.append(self)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit_called = True
        return df.fillna(1)


@pytest.fixture(autouse=True)
def patch_pipeline_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeMiceImputer.created = []
    FakeMiceImputer.next_report = {"severity": {"lambda": 0.1}}
    FakeRegresionImputer.created = []

    monkeypatch.setattr(pipeline_module, "MiceImputer", FakeMiceImputer)
    monkeypatch.setattr(pipeline_module, "RegresionImputer", FakeRegresionImputer)


def make_df(rows: int, cols: int, missing_in_first_col: int) -> pd.DataFrame:
    data = {
        f"x{i}": [float(row + i) for row in range(rows)]
        for i in range(cols)
    }
    df = pd.DataFrame(data)
    if missing_in_first_col:
        df.loc[: missing_in_first_col - 1, "x0"] = None
    return df


def test_inference_with_few_missing_values_uses_mice() -> None:
    df = make_df(rows=25, cols=2, missing_in_first_col=1)

    result = ImputationPipeline().run(df, goal="inference")

    assert result["decision"] == "mice"
    assert FakeMiceImputer.created[0].m == 5
    assert FakeRegresionImputer.created == []
    assert result["imputer_report"] == {"severity": {"lambda": 0.1}}
    assert result["warnings"] == []
    assert "diagnostico" not in result
    assert result["criteria"] == {
        "max_pct": 4.0,
        "ratio": 12.5,
        "goal": "inference",
    }
    assert "Se eligio MICE con m=5" in result["reasoning"]
    assert "hacer inferencia estadistica" in result["reasoning"]
    assert "goal=" not in result["reasoning"]


def test_pipeline_passes_beta_vars_to_mice() -> None:
    df = make_df(rows=25, cols=2, missing_in_first_col=1)

    ImputationPipeline(beta_vars=["x0"]).run(df, goal="inference")

    assert FakeMiceImputer.created[0].beta_vars == ["x0"]


def test_prediction_with_less_than_five_percent_missing_uses_stochastic_regression() -> None:
    df = make_df(rows=25, cols=2, missing_in_first_col=1)

    result = ImputationPipeline(beta_vars=["x0"]).run(df, goal="prediction")

    assert result["decision"] == "regresion_estocastica"
    assert FakeRegresionImputer.created[0].method == "stochastic_regression"
    assert FakeMiceImputer.created == []
    assert result["imputer_report"] is None
    assert "Se eligio regresion estocastica" in result["reasoning"]
    assert "goal=" not in result["reasoning"]


def test_prediction_with_high_missing_uses_mice_m_10() -> None:
    df = make_df(rows=20, cols=2, missing_in_first_col=3)

    result = ImputationPipeline().run(df, goal="prediction")

    assert result["decision"] == "mice"
    assert FakeMiceImputer.created[0].m == 10
    assert "m=10 (mayor numero de imputaciones)" in result["reasoning"]


def test_prediction_with_intermediate_missing_and_good_ratio_uses_mice_m_5() -> None:
    df = make_df(rows=20, cols=2, missing_in_first_col=2)

    result = ImputationPipeline().run(df, goal="prediction")

    assert result["decision"] == "mice"
    assert FakeMiceImputer.created[0].m == 5


def test_mice_lambda_over_threshold_adds_warning() -> None:
    FakeMiceImputer.next_report = {
        "severity": {
            "lambda": {
                "x": 0.31,
                "y": 0.2,
            }
        }
    }
    df = make_df(rows=20, cols=2, missing_in_first_col=2)

    result = ImputationPipeline().run(df, goal="prediction")

    assert result["warnings"] == [
        "Lambda de severidad supero el umbral 0.30 para: x."
    ]


def test_mice_scalar_lambda_over_threshold_warns_global() -> None:
    FakeMiceImputer.next_report = {"severity": {"lambda": 0.35}}
    df = make_df(rows=20, cols=2, missing_in_first_col=2)

    result = ImputationPipeline().run(df, goal="prediction")

    assert result["warnings"] == [
        "Lambda de severidad supero el umbral 0.30 para: global."
    ]
