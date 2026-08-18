from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import core.pipeline as pipeline_module
from core.pipeline import ImputationPipeline


class FakeDiagnosticoRunner:
    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        return {"diagnostico": "ok", "rows": len(df)}


class FakeMiceImputer:
    created: list["FakeMiceImputer"] = []
    next_report: dict[str, Any] = {
        "severity": {"lambda": 0.1},
    }

    def __init__(self, m: int = 5) -> None:
        self.m = m
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

    monkeypatch.setattr(pipeline_module, "DiagnosticoRunner", FakeDiagnosticoRunner)
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


def test_prediction_with_less_than_five_percent_missing_uses_stochastic_regression() -> None:
    df = make_df(rows=25, cols=2, missing_in_first_col=1)

    result = ImputationPipeline().run(df, goal="prediction")

    assert result["decision"] == "regresion_estocastica"
    assert FakeRegresionImputer.created[0].method == "stochastic_regression"
    assert FakeMiceImputer.created == []
    assert result["imputer_report"] is None


def test_prediction_with_high_missing_uses_mice_m_10() -> None:
    df = make_df(rows=20, cols=2, missing_in_first_col=3)

    result = ImputationPipeline().run(df, goal="prediction")

    assert result["decision"] == "mice"
    assert FakeMiceImputer.created[0].m == 10


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
