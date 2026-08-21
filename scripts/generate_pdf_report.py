from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_pdf(report_json_path: Path, output_pdf_path: Path) -> None:
    with open(report_json_path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    _add_cover(story, styles, report)
    story.append(PageBreak())
    _add_exploration(story, styles, report)
    story.append(PageBreak())
    _add_decision(story, styles, report)
    _add_imputation(story, styles, report)
    _add_assumptions(story, styles, report)
    _add_warnings(story, styles, report)

    doc.build(story)


def _add_cover(story: list[Any], styles: dict[str, Any], report: dict[str, Any]) -> None:
    metadata = report.get("metadata", {})
    story.append(Paragraph("Informe de Imputación de Datos Faltantes", styles["Title"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Dataset: {_text(metadata.get('dataset_name', 'dataset'))}", styles["Normal"]))
    story.append(Paragraph(f"Fecha: {_text(metadata.get('fecha_generacion'))}", styles["Normal"]))
    story.append(
        Paragraph(
            f"Dimensiones: {_text(metadata.get('n_rows'))} x {_text(metadata.get('n_columns'))}",
            styles["Normal"],
        )
    )


def _add_exploration(story: list[Any], styles: dict[str, Any], report: dict[str, Any]) -> None:
    exploration = report.get("exploracion_previa") or {}
    story.append(Paragraph("Exploración Previa", styles["Heading1"]))

    missing_rows = _missing_summary_rows(exploration)
    if missing_rows:
        story.append(_table([["Variable", "% faltantes"]] + missing_rows))
    else:
        story.append(Paragraph("No aplica", styles["Normal"]))

    story.append(Spacer(1, 12))
    if exploration.get("numeric_analysis_skipped"):
        skipped = exploration["numeric_analysis_skipped"]
        story.append(
            Paragraph(
                f"Análisis numérico omitido: {_text(skipped.get('reason'))}",
                styles["Normal"],
            )
        )
    else:
        _add_key_value_block(
            story,
            styles,
            "Test de Little (naniar)",
            exploration.get("mcar_test_naniar"),
        )
        _add_key_value_block(
            story,
            styles,
            "Test de Little (manual)",
            exploration.get("little_test_manual"),
        )
        _add_key_value_block(
            story,
            styles,
            "Estimaciones EM manuales",
            exploration.get("em_estimates_manual"),
        )
        _add_key_value_block(
            story,
            styles,
            "Estimaciones EM mvnmle",
            exploration.get("em_estimates_mvnmle"),
        )


def _add_decision(story: list[Any], styles: dict[str, Any], report: dict[str, Any]) -> None:
    decision = report.get("decision") or {}
    story.append(Spacer(1, 18))
    story.append(Paragraph("Decisión del Método", styles["Heading1"]))
    story.append(Paragraph(f"Método: {_text(decision.get('metodo', 'No aplica'))}", styles["Normal"]))
    story.append(Paragraph(_text(decision.get("razonamiento", "No aplica")), styles["Normal"]))


def _add_imputation(story: list[Any], styles: dict[str, Any], report: dict[str, Any]) -> None:
    story.append(Spacer(1, 18))
    story.append(Paragraph("Resultado de la Imputación", styles["Heading1"]))
    imputation = report.get("imputacion")
    method = (report.get("decision") or {}).get("metodo")

    if imputation is None:
        story.append(Paragraph("No aplica", styles["Normal"]))
        return
    if method != "mice":
        story.append(Paragraph("No aplica pooling de Rubin.", styles["Normal"]))
        return

    rows = [["Campo", "Valor"]]
    rows.extend(_dict_rows("point_estimate", imputation.get("point_estimate")))
    rows.extend(_dict_rows("severity", imputation.get("severity")))
    story.append(_table(rows if len(rows) > 1 else [["Campo", "Valor"], ["No aplica", ""]]))


def _add_assumptions(story: list[Any], styles: dict[str, Any], report: dict[str, Any]) -> None:
    story.append(Spacer(1, 18))
    story.append(Paragraph("Cumplimiento de Supuestos", styles["Heading1"]))
    imputation = report.get("imputacion") or {}
    assumptions = imputation.get("assumptions") if isinstance(imputation, dict) else None

    if not assumptions:
        story.append(Paragraph("No aplica", styles["Normal"]))
        return

    rows, failing_indexes = _assumption_rows(assumptions)
    if len(rows) == 1:
        story.append(Paragraph("No aplica", styles["Normal"]))
        return

    style_commands: list[tuple[Any, ...]] = []
    for row_index in failing_indexes:
        style_commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.mistyrose))
    story.append(_table(rows, style_commands))


def _add_warnings(story: list[Any], styles: dict[str, Any], report: dict[str, Any]) -> None:
    story.append(Spacer(1, 18))
    story.append(Paragraph("Advertencias", styles["Heading1"]))
    warnings = report.get("advertencias") or []
    if not warnings:
        story.append(Paragraph("Sin advertencias", styles["Normal"]))
        return
    for warning in warnings:
        story.append(Paragraph(f"• {_text(warning)}", styles["Normal"]))


def _missing_summary_rows(exploration: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    summaries = (
        exploration.get("md_pattern_summary", {})
        .get("miss_var_summary", [])
    )
    if isinstance(summaries, list):
        for entry in summaries:
            if not isinstance(entry, dict):
                continue
            variable = entry.get("variable") or entry.get("var") or entry.get("feature")
            pct = entry.get("pct_miss") or entry.get("pct_missing") or entry.get("missing_percent")
            if variable is not None and pct is not None:
                rows.append([_text(variable), _text(pct)])
    return rows


def _add_key_value_block(
    story: list[Any],
    styles: dict[str, Any],
    title: str,
    payload: Any,
) -> None:
    story.append(Paragraph(title, styles["Heading2"]))
    if not payload:
        story.append(Paragraph("No aplica", styles["Normal"]))
        return
    if isinstance(payload, dict):
        story.append(_table([["Campo", "Valor"]] + _dict_rows("", payload)))
    else:
        story.append(Paragraph(_text(payload), styles["Normal"]))


def _dict_rows(prefix: str, payload: Any) -> list[list[str]]:
    if not isinstance(payload, dict):
        return []
    rows: list[list[str]] = []
    for key, value in payload.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_dict_rows(label, value))
        else:
            rows.append([label, _text(value)])
    return rows


def _assumption_rows(assumptions: dict[str, Any]) -> tuple[list[list[str]], list[int]]:
    rows = [["Variable", "Supuesto", "Prueba/Detalle", "Resultado", "Cumple"]]
    failing_indexes: list[int] = []

    convergence = assumptions.get("convergence", {})
    if isinstance(convergence, dict):
        for variable, payload in convergence.items():
            if not isinstance(payload, dict):
                continue
            converged_value = payload.get("converged")
            row = [
                _text(variable),
                "convergence",
                "Convergencia de cadenas MICE",
                "Convergio" if converged_value is True else ("No convergio" if converged_value is False else "No aplica"),
                _text(converged_value),
            ]
            rows.append(row)
            if payload.get("converged") is False:
                failing_indexes.append(len(rows) - 1)

    distribution = assumptions.get("distribution_comparison", {})
    if isinstance(distribution, dict):
        for variable, payload in distribution.items():
            _append_test_row(
                rows,
                failing_indexes,
                variable,
                "distribution_comparison",
                payload,
                default_test_label="Anderson-Darling (k-muestras)",
            )

    for variable, payload in assumptions.items():
        if variable in ("convergence", "distribution_comparison"):
            continue
        if not isinstance(payload, dict):
            continue
        _append_test_row(rows, failing_indexes, variable, "normality", payload.get("normality"))
        _append_test_row(
            rows,
            failing_indexes,
            variable,
            "homoscedasticity",
            payload.get("homoscedasticity"),
        )

    return rows, failing_indexes


def _append_test_row(
    rows: list[list[str]],
    failing_indexes: list[int],
    variable: str,
    assumption_name: str,
    payload: Any,
    default_test_label: str = "No aplica",
) -> None:
    if not isinstance(payload, dict):
        return
    rows.append(
        [
            _text(variable),
            assumption_name,
            _text(payload.get("test_used", default_test_label)),
            f"stat={_text(payload.get('statistic', payload.get('ad_statistic')))}; p={_text(payload.get('p_value'))}",
            _text(payload.get("meets_assumption")),
        ]
    )
    if payload.get("meets_assumption") is False:
        failing_indexes.append(len(rows) - 1)


def _table(rows: list[list[str]], extra_style: list[tuple[Any, ...]] | None = None) -> Table:
    table = Table(rows, repeatRows=1)
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if extra_style:
        commands.extend(extra_style)
    table.setStyle(TableStyle(commands))
    return table


def _text(value: Any) -> str:
    if value is None:
        return "No aplica"
    return str(value)
