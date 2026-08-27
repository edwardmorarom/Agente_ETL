from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from scripts.generate_html_report import _text, _to_latex_label, generate_html


def sample_report() -> dict[str, Any]:
    return {
        "metadata": {
            "dataset_name": "demo",
            "fecha_generacion": "2026-08-20T00:00:00",
            "n_rows": 3,
            "n_columns": 2,
        },
        "exploracion_previa": {
            "md_pattern_summary": {
                "miss_var_summary": [
                    {"variable": "x", "pct_miss": 33.333333},
                ]
            },
            "comparison_plots": {
                "x": {
                    "boxplot": '<div id="plotly-box">box</div>',
                    "histograma": '<div id="plotly-hist">hist</div>',
                    "qqplot": '<div id="plotly-qq">qq</div>',
                }
            },
            "em_estimates_manual": {
                "mu": {"Ladder": 5.123456},
                "sigma": {"Ladder": {"LGDP": 1.234567}},
            },
        },
        "resumen_descriptivo": {
            "x": {
                "antes": {
                    "media": 1.111111,
                    "mediana": 1.0,
                    "desviacion_estandar": 0.123456,
                    "minimo": 1.0,
                    "maximo": 2.0,
                },
                "despues": {
                    "media": 1.222222,
                    "mediana": 1.2,
                    "desviacion_estandar": 0.223456,
                    "minimo": 1.0,
                    "maximo": 2.0,
                },
            }
        },
        "decision": {
            "metodo": "mice",
            "razonamiento": "razon",
            "criterios": {"max_pct": 33.33, "ratio": 1.5, "goal": "inference"},
        },
        "imputacion": {
            "point_estimate": {f"x{i}": float(i) for i in range(12)},
            "severity": {"lambda": 0.123456, "r": 0.2, "df": 10.0, "gamma": 0.3},
            "beta_distribution_fit": {
                "x": {
                    "scale_used": "percentage_0_100",
                    "n_imputations": 5,
                    "shape1_pooled": 2.1,
                    "shape2_pooled": 3.2,
                    "covariance_pooled": [[0.1, 0.01], [0.01, 0.2]],
                    "severity_per_parameter": {
                        "shape1": {
                            "lambda": 0.11,
                            "r": 0.12,
                            "df": 20.0,
                            "gamma": 0.13,
                        },
                        "shape2": {
                            "lambda": 0.21,
                            "r": 0.22,
                            "df": 21.0,
                            "gamma": 0.23,
                        },
                    },
                    "severity_joint": {
                        "lambda": 0.31,
                        "r": 0.32,
                        "df": 22.0,
                        "gamma": 0.33,
                    },
                }
            },
        },
        "advertencias": [],
        "interpretacion_ia": {
            "general": "Resumen ejecutivo IA.",
            "exploracion": "Nota IA exploracion.",
            "resumen_descriptivo": "Nota IA resumen.",
            "decision": "Nota IA decision.",
            "imputacion": "Nota IA imputacion.",
            "supuestos": "Nota IA supuestos.",
            "advertencias": "Nota IA advertencias.",
        },
    }


def test_text_rounds_float_to_four_decimals() -> None:
    assert _text(5.123456) == "5.1235"
    assert _text(5.0) == "5"


def test_to_latex_label_converts_sigma() -> None:
    assert _to_latex_label("sigma.Ladder.LGDP") == "\\sigma_{Ladder,LGDP}"


def test_html_contains_variable_selector_and_scrollable_table(tmp_path: Path) -> None:
    report_json = tmp_path / "report.json"
    output_html = tmp_path / "report.html"
    report_json.write_text(json.dumps(sample_report()), encoding="utf-8")

    generate_html(report_json, output_html)
    html = output_html.read_text(encoding="utf-8")

    assert "var-selector-btn" in html
    assert "table-scroll" in html
    assert "1.1111" in html
    assert '<div id="plotly-box">box</div>' in html
    assert '<img src="data:image/png;base64' not in html
    assert "Point Estimate" in html
    assert "Severidad" in html
    assert "Ajuste Beta" in html
    assert "\\alpha = 2.1" in html
    assert "\\beta = 3.2" in html
    assert 'href="#interpretacion"' in html
    assert 'section id="interpretacion" class="ai-section"' in html
    assert "<p>Resumen ejecutivo IA.</p>" in html
    assert "Nota IA exploracion." in html
    assert "Nota IA resumen." in html
    assert "Nota IA decision." in html
    assert "Nota IA imputacion." in html
    assert "Nota IA supuestos." in html
    assert "Nota IA advertencias." in html
    assert html.count('class="ai-note"') == 6
    assert html.index('href="#interpretacion"') < html.index('href="#exploracion"')
    assert html.index('id="interpretacion"') < html.index('id="exploracion"')


def test_html_shows_ai_interpretation_fallback(tmp_path: Path) -> None:
    report = sample_report()
    report["interpretacion_ia"] = None
    report_json = tmp_path / "report.json"
    output_html = tmp_path / "report.html"
    report_json.write_text(json.dumps(report), encoding="utf-8")

    generate_html(report_json, output_html)
    html = output_html.read_text(encoding="utf-8")

    assert "Interpretacion de IA no disponible para esta corrida." in html
    assert 'class="ai-note"' not in html
