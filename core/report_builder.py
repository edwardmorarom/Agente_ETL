from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def build_report_json(
    diagnostico: dict[str, Any],
    pipeline_result: dict[str, Any],
    dataset_name: str = "dataset",
    output_path: Path | None = None,
) -> dict[str, Any]:
    if output_path is not None:
        output_path = Path(output_path)

    n_rows, n_columns = _extract_dimensions(diagnostico, pipeline_result)
    report = {
        "metadata": {
            "dataset_name": dataset_name,
            "fecha_generacion": datetime.now().isoformat(),
            "n_rows": n_rows,
            "n_columns": n_columns,
        },
        "exploracion_previa": diagnostico,
        "decision": {
            "metodo": pipeline_result["decision"],
            "razonamiento": pipeline_result["reasoning"],
        },
        "imputacion": pipeline_result.get("imputer_report"),
        "advertencias": pipeline_result.get("warnings", []),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False, default=str)

    return report


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
