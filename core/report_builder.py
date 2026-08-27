from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_report_json(
    diagnostico: dict[str, Any],
    pipeline_result: dict[str, Any],
    dataset_name: str = "dataset",
    output_path: Path | str | None = None,
    df_original: pd.DataFrame | None = None,
    comparison_plots: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if output_path is not None:
        output_path = Path(output_path)

    n_rows, n_columns = _extract_dimensions(diagnostico, pipeline_result)
    exploracion_previa = deepcopy(diagnostico)
    if comparison_plots is not None:
        exploracion_previa["comparison_plots"] = comparison_plots

    report = {
        "metadata": {
            "dataset_name": dataset_name,
            "fecha_generacion": datetime.now().isoformat(),
            "n_rows": n_rows,
            "n_columns": n_columns,
        },
        "exploracion_previa": exploracion_previa,
        "decision": {
            "metodo": pipeline_result["decision"],
            "razonamiento": pipeline_result["reasoning"],
            "criterios": pipeline_result.get("criteria"),
        },
        "imputacion": pipeline_result.get("imputer_report"),
        "advertencias": pipeline_result.get("warnings", []),
    }
    if df_original is not None:
        report["resumen_descriptivo"] = build_descriptive_summary(
            df_original,
            pipeline_result.get("imputed_data"),
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False, default=str)

    return report


def describe_series(s: pd.Series) -> dict[str, float | int] | None:
    s_clean = s.dropna()
    if s_clean.empty:
        return None
    return {
        "n": int(s_clean.count()),
        "media": float(s_clean.mean()),
        "desviacion_estandar": float(s_clean.std()),
        "minimo": float(s_clean.min()),
        "q1": float(s_clean.quantile(0.25)),
        "mediana": float(s_clean.median()),
        "q3": float(s_clean.quantile(0.75)),
        "maximo": float(s_clean.max()),
    }


def build_descriptive_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    numeric_columns = df_before.select_dtypes(include="number").columns

    for col in numeric_columns:
        summary[col] = {
            "antes": describe_series(df_before[col]),
            "despues": (
                describe_series(df_after[col])
                if df_after is not None and col in df_after.columns
                else None
            ),
        }

    return summary


def _extract_dimensions(
    diagnostico: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> tuple[int | None, int | None]:
    n_rows = _first_present(
        diagnostico,
        ["n_rows", "metadata.n_rows", "profile.n_rows", "perfil.n_rows"],
    )
    n_columns = _first_present(
        diagnostico,
        ["n_columns", "metadata.n_columns", "profile.n_columns", "perfil.n_columns"],
    )

    if (n_rows is None or n_columns is None) and "imputed_data" in pipeline_result:
        shape = getattr(pipeline_result["imputed_data"], "shape", None)
        if shape is not None and len(shape) == 2:
            n_rows = n_rows if n_rows is not None else int(shape[0])
            n_columns = n_columns if n_columns is not None else int(shape[1])

    return _safe_int(n_rows), _safe_int(n_columns)


def _first_present(payload: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        current: Any = payload
        found = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found:
            return current
    return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
