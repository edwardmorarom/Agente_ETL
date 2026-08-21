from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.report_builder import build_report_json


def sample_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostico = {
        "n_rows": 2,
        "n_columns": 2,
        "md_pattern_summary": {
            "miss_var_summary": [
                {"variable": "x", "pct_miss": 50.0},
            ]
        },
    }
    pipeline_result = {
        "decision": "mice",
        "reasoning": "Se eligio MICE por inferencia.",
        "imputed_data": pd.DataFrame({"x": ["DATO_SECRETO"], "y": [1.0]}),
        "imputer_report": {
            "case": "multivariate",
            "severity": {"lambda": 0.1},
        },
        "warnings": ["warning de prueba"],
    }
    return diagnostico, pipeline_result


def test_build_report_json_never_includes_imputed_data_or_dataframe_contents() -> None:
    diagnostico, pipeline_result = sample_inputs()

    report = build_report_json(diagnostico, pipeline_result, dataset_name="demo")
    report_text = str(report)

    assert "imputed_data" not in report_text
    assert "DATO_SECRETO" not in report_text


def test_build_report_json_has_expected_structure() -> None:
    diagnostico, pipeline_result = sample_inputs()

    report = build_report_json(diagnostico, pipeline_result, dataset_name="demo")

    assert set(report) == {
        "metadata",
        "exploracion_previa",
        "decision",
        "imputacion",
        "advertencias",
    }
    assert report["metadata"]["dataset_name"] == "demo"
    assert report["metadata"]["n_rows"] == 2
    assert report["metadata"]["n_columns"] == 2
    assert "fecha_generacion" in report["metadata"]
    assert report["exploracion_previa"] == diagnostico
    assert report["decision"] == {
        "metodo": "mice",
        "razonamiento": "Se eligio MICE por inferencia.",
    }
    assert report["imputacion"] == pipeline_result["imputer_report"]
    assert report["advertencias"] == ["warning de prueba"]


def test_build_report_json_accepts_output_path_as_string(tmp_path: Path) -> None:
    diagnostico, pipeline_result = sample_inputs()
    output_path = str(tmp_path / "reporte.json")

    report = build_report_json(
        diagnostico,
        pipeline_result,
        dataset_name="demo",
        output_path=output_path,
    )

    with open(output_path, "r", encoding="utf-8") as report_file:
        written = json.load(report_file)

    assert written["metadata"]["dataset_name"] == "demo"
    assert written["decision"] == report["decision"]
