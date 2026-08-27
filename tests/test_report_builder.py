from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.report_builder import build_descriptive_summary, build_report_json


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
        "criteria": {"max_pct": 50.0, "ratio": 1.0, "goal": "inference"},
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
        "criterios": {"max_pct": 50.0, "ratio": 1.0, "goal": "inference"},
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


def test_build_descriptive_summary_contains_only_aggregated_fields() -> None:
    df_before = pd.DataFrame(
        {
            "x": [10.0, 20.0, None, 40.0],
            "category": ["a", "b", "c", "d"],
        }
    )
    df_after = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0, 40.0],
            "category": ["a", "b", "c", "d"],
        }
    )

    summary = build_descriptive_summary(df_before, df_after)
    expected_fields = {
        "n",
        "media",
        "desviacion_estandar",
        "minimo",
        "q1",
        "mediana",
        "q3",
        "maximo",
    }

    assert set(summary) == {"x"}
    assert set(summary["x"]["antes"]) == expected_fields
    assert set(summary["x"]["despues"]) == expected_fields
    assert "category" not in summary
    assert "10.0, 20.0" not in str(summary)


def test_build_report_json_adds_descriptive_summary_without_rows() -> None:
    diagnostico, pipeline_result = sample_inputs()
    df_original = pd.DataFrame({"x": [10.0, None], "label": ["a", "b"]})
    pipeline_result["imputed_data"] = pd.DataFrame(
        {"x": [10.0, 20.0], "label": ["a", "b"]}
    )

    report = build_report_json(
        diagnostico,
        pipeline_result,
        dataset_name="demo",
        df_original=df_original,
    )

    assert "resumen_descriptivo" in report
    assert set(report["resumen_descriptivo"]) == {"x"}
    assert "label" not in report["resumen_descriptivo"]
    assert "imputed_data" not in str(report)


def test_build_report_json_adds_embedded_comparison_plots() -> None:
    diagnostico, pipeline_result = sample_inputs()
    comparison_plots = {
        "x": {
            "boxplot": "<div>box</div>",
            "histograma": "<div>hist</div>",
            "qqplot": "<div>qq</div>",
        }
    }

    report = build_report_json(
        diagnostico,
        pipeline_result,
        dataset_name="demo",
        comparison_plots=comparison_plots,
    )

    assert report["exploracion_previa"]["comparison_plots"] == comparison_plots
