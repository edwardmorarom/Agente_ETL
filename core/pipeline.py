from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from imputers.diagnostico_runner import DiagnosticoRunner
from imputers.mice_imputer import MiceImputer
from imputers.regresion_imputer import RegresionImputer


class ImputationPipeline:
    def __init__(
        self,
        low_missing_threshold: float = 5.0,
        high_missing_threshold: float = 15.0,
        low_ratio_threshold: float = 5.0,
        lambda_warning_threshold: float = 0.30,
    ) -> None:
        self.low_missing_threshold = low_missing_threshold
        self.high_missing_threshold = high_missing_threshold
        self.low_ratio_threshold = low_ratio_threshold
        self.lambda_warning_threshold = lambda_warning_threshold

    def run(
        self,
        df: pd.DataFrame,
        goal: Literal["inference", "prediction"],
    ) -> dict[str, Any]:
        diagnostico = DiagnosticoRunner().run(df)
        max_pct = float((df.isna().mean() * 100).max()) if df.shape[1] else 0.0
        ratio = len(df) / df.shape[1] if df.shape[1] else float("inf")

        decision, imputer = self._select_imputer(max_pct, ratio, goal)
        imputed_data = imputer.fit_transform(df)

        imputer_report = None
        warnings: list[str] = []
        if isinstance(imputer, MiceImputer):
            imputer_report = imputer.last_report
            warnings = self._build_lambda_warnings(imputer_report)

        return {
            "decision": decision,
            "reasoning": self._build_reasoning(decision, max_pct, ratio, goal),
            "diagnostico": diagnostico,
            "imputed_data": imputed_data,
            "imputer_report": imputer_report,
            "warnings": warnings,
        }

    def _select_imputer(
        self,
        max_pct: float,
        ratio: float,
        goal: Literal["inference", "prediction"],
    ) -> tuple[Literal["mice", "regresion_estocastica"], MiceImputer | RegresionImputer]:
        needs_stronger_mice = (
            max_pct >= self.high_missing_threshold
            or ratio <= self.low_ratio_threshold
        )

        if goal == "inference":
            m = 10 if needs_stronger_mice else 5
            return "mice", MiceImputer(m=m)

        if max_pct < self.low_missing_threshold:
            return "regresion_estocastica", RegresionImputer(
                method="stochastic_regression"
            )

        m = 10 if needs_stronger_mice else 5
        return "mice", MiceImputer(m=m)

    def _build_reasoning(
        self,
        decision: str,
        max_pct: float,
        ratio: float,
        goal: Literal["inference", "prediction"],
    ) -> str:
        return (
            f"Se eligio {decision} con goal={goal}, "
            f"max_pct={max_pct:.2f} y ratio={ratio:.2f}."
        )

    def _build_lambda_warnings(self, report: dict[str, Any] | None) -> list[str]:
        if not report:
            return []

        severity = report.get("severity")
        if not isinstance(severity, dict) or "lambda" not in severity:
            return []

        lambda_value = severity["lambda"]
        exceeded = self._lambda_values_over_threshold(lambda_value)
        if not exceeded:
            return []

        variables = ", ".join(exceeded)
        return [
            "Lambda de severidad supero el umbral "
            f"{self.lambda_warning_threshold:.2f} para: {variables}."
        ]

    def _lambda_values_over_threshold(self, lambda_value: Any) -> list[str]:
        if isinstance(lambda_value, dict):
            return [
                str(variable)
                for variable, value in lambda_value.items()
                if self._is_over_lambda_threshold(value)
            ]

        if self._is_over_lambda_threshold(lambda_value):
            return ["global"]

        return []

    def _is_over_lambda_threshold(self, value: Any) -> bool:
        try:
            return float(value) >= self.lambda_warning_threshold
        except (TypeError, ValueError):
            return False
