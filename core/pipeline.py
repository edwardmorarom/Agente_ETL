from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from imputers.mice_imputer import MiceImputer
from imputers.regresion_imputer import RegresionImputer


class ImputationPipeline:
    def __init__(
        self,
        low_missing_threshold: float = 5.0,
        high_missing_threshold: float = 15.0,
        low_ratio_threshold: float = 5.0,
        lambda_warning_threshold: float = 0.30,
        mice_vars: list[str] | None = None,
        beta_vars: list[str] | None = None,
    ) -> None:
        self.low_missing_threshold = low_missing_threshold
        self.high_missing_threshold = high_missing_threshold
        self.low_ratio_threshold = low_ratio_threshold
        self.lambda_warning_threshold = lambda_warning_threshold
        self.mice_vars = mice_vars
        self.beta_vars = beta_vars

    def run(
        self,
        df: pd.DataFrame,
        goal: Literal["inference", "prediction"],
    ) -> dict[str, Any]:
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
            "criteria": {
                "max_pct": round(max_pct, 2),
                "ratio": round(ratio, 2),
                "goal": goal,
            },
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
            return "mice", MiceImputer(
                vars=self.mice_vars,
                m=m,
                beta_vars=self.beta_vars,
            )

        if max_pct < self.low_missing_threshold:
            return "regresion_estocastica", RegresionImputer(
                method="stochastic_regression"
            )

        m = 10 if needs_stronger_mice else 5
        return "mice", MiceImputer(
            vars=self.mice_vars,
            m=m,
            beta_vars=self.beta_vars,
        )

    def _build_reasoning(
        self,
        decision: str,
        max_pct: float,
        ratio: float,
        goal: Literal["inference", "prediction"],
    ) -> str:
        goal_text = (
            "hacer inferencia estadistica"
            if goal == "inference"
            else "obtener un valor predictivo rapido"
        )

        if decision == "regresion_estocastica":
            return (
                "Se eligio regresion estocastica porque la variable con mas "
                f"datos faltantes tiene solo {max_pct:.1f}% de valores vacios, "
                "un nivel bajo donde la perdida de informacion es minima y no "
                "justifica el costo computacional de metodos mas complejos."
            )

        needs_stronger_mice = (
            max_pct >= self.high_missing_threshold
            or ratio <= self.low_ratio_threshold
        )
        m_text = (
            "m=10 (mayor numero de imputaciones)"
            if needs_stronger_mice
            else "m=5"
        )
        razon_intensidad = (
            f"ya que el {max_pct:.1f}% de faltantes en la peor variable supera "
            f"el {self.high_missing_threshold:.0f}% o hay pocas observaciones "
            f"por variable (ratio={ratio:.1f})"
            if needs_stronger_mice
            else f"dado un nivel moderado de {max_pct:.1f}% de faltantes y "
            f"suficientes observaciones por variable (ratio={ratio:.1f})"
        )
        return (
            f"Se eligio MICE con {m_text} para {goal_text}, {razon_intensidad}. "
            "MICE permite capturar la incertidumbre de la imputacion mediante "
            "las Reglas de Rubin, algo que un metodo simple no ofrece."
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
