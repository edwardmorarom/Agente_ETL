from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import re
from typing import Any

from plotly.offline import get_plotlyjs


FRIENDLY_PLOT_NAMES = {
    "gg_miss_var.png": "Faltantes por variable",
    "gg_miss_upset.png": "Patrones de combinaciones de faltantes",
    "aggr.png": "Agregado de faltantes (VIM)",
    "plot_missing.png": "Mapa de calor de faltantes",
    "histMiss.png": "Histograma de faltantes",
    "barMiss.png": "Barras de faltantes por variable",
}


def generate_html(report_json_path: Path, output_html_path: Path) -> None:
    with open(report_json_path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(_render_dashboard(report), encoding="utf-8")


def _load_katex_assets() -> tuple[str, str]:
    katex_dir = Path(__file__).resolve().parents[1] / "assets" / "katex"
    js_path = katex_dir / "katex.min.js"
    css_path = katex_dir / "katex.min.css"
    fonts_dir = katex_dir / "fonts"

    js_content = js_path.read_text(encoding="utf-8") if js_path.exists() else ""
    css_content = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    def replace_font(match: re.Match[str]) -> str:
        font_name = match.group(1)
        font_path = fonts_dir / font_name
        if not font_path.exists():
            return match.group(0)
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return f"url(data:font/woff2;base64,{encoded})"

    css_content = re.sub(
        r"url\(['\"]?fonts/([^)'\"]+\.woff2)['\"]?\)",
        replace_font,
        css_content,
    )
    return js_content, css_content


def _render_dashboard(report: dict[str, Any]) -> str:
    metadata = report.get("metadata") or {}
    decision = report.get("decision") or {}
    warnings = report.get("advertencias") or []
    method = decision.get("metodo", "No aplica")
    warning_count = len(warnings)
    katex_js, katex_css = _load_katex_assets()
    plotly_js = get_plotlyjs()
    beta_section = _section_beta_distribution_fit(report)
    interpretation = report.get("interpretacion_ia")

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Informe de Imputación de Datos Faltantes</title>
  <style>{katex_css}</style>
  <script>{katex_js}</script>
  <script>{plotly_js}</script>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d9e2ec;
      --blue: #2563eb;
      --green: #15803d;
      --red: #b42318;
      --red-soft: #fde2e1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, 'Segoe UI', sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(135deg, #1e3a8a, #2563eb);
      color: white;
      padding: 36px clamp(18px, 5vw, 56px);
    }}
    header h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 44px); }}
    header p {{ margin: 4px 0; opacity: 0.92; }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      gap: 10px;
      overflow-x: auto;
      padding: 12px clamp(18px, 5vw, 56px);
      background: rgba(255, 255, 255, 0.95);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    nav a {{
      color: var(--blue);
      text-decoration: none;
      font-weight: 600;
      white-space: nowrap;
      padding: 8px 16px;
      border-radius: 999px;
      background: #eef4ff;
      transition: background 0.15s, color 0.15s;
    }}
    nav a:hover {{ background: var(--blue); color: white; }}
    nav a.active {{ background: #1e3a8a; color: white; }}
    main {{ padding: 24px clamp(18px, 5vw, 56px) 56px; }}
    section {{
      margin: 24px 0;
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }}
    h2 {{ margin-top: 0; color: #172554; }}
    h3 {{ margin-bottom: 10px; color: #1e3a8a; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 22px;
    }}
    .kpi {{
      background: rgba(255, 255, 255, 0.13);
      border: 1px solid rgba(255, 255, 255, 0.25);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.16);
    }}
    .kpi .label {{ display: block; opacity: 0.84; font-size: 14px; }}
    .kpi .value {{ display: block; margin-top: 8px; font-size: 24px; font-weight: 800; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      color: white;
      font-weight: 700;
    }}
    .badge.mice {{ background: var(--blue); }}
    .badge.regresion_estocastica {{ background: var(--green); }}
    .badge.ok {{ background: var(--green); }}
    .badge.alert {{ background: var(--red); }}
    .ai-section {{
      background: linear-gradient(135deg, #eef4ff, #f5efff);
      border-color: #c7d2fe;
    }}
    .ai-title {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .ai-badge {{
      background: #6d28d9;
      color: white;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .ai-content p {{
      margin: 0 0 12px;
      color: #26354d;
      font-size: 16px;
    }}
    .ai-content p:last-child {{ margin-bottom: 0; }}
    .ai-note {{
      margin-top: 18px;
      padding: 14px 16px;
      background: #f5efff;
      border-left: 4px solid #6d28d9;
      border-radius: 8px;
      color: #344054;
    }}
    .ai-note strong {{ color: #4c1d95; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      background: white;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef4ff; color: #1e3a8a; }}
    .fila-alerta td {{ background: var(--red-soft); }}
    .table-scroll {{
      max-height: 360px;
      overflow-y: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      margin: 12px 0 18px;
    }}
    .table-scroll table {{ margin: 0; }}
    .table-scroll thead th {{ position: sticky; top: 0; background: #eef4ff; z-index: 1; }}
    .plots, .comparison-plots {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .plot-card, .summary-card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      background: #fbfdff;
    }}
    .plot-card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
      border: 1px solid var(--line);
    }}
    .selector {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .var-selector-btn {{
      border: 1px solid #bfdbfe;
      background: #eef4ff;
      color: var(--blue);
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
      font-weight: 700;
    }}
    .var-selector-btn.active {{ background: var(--blue); color: white; }}
    .toggle-panel {{ display: none; }}
    .toggle-panel.active {{ display: block; }}
    .variable-summary {{ display: none; }}
    .variable-summary.active {{ display: block; }}
    .summary-compare {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
    }}
    .metric-row:last-child {{ border-bottom: 0; }}
    .decision-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      background: #fbfdff;
    }}
    .method-row {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .method-badge {{
      font-size: 22px;
      padding: 10px 16px;
      border-radius: 12px;
    }}
    .criteria-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat-pill {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: #eef4ff;
    }}
    .stat-pill .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .stat-pill .value {{
      display: block;
      color: #172554;
      font-weight: 800;
    }}
    .decision-reasoning {{
      margin: 0;
      color: #344054;
      font-size: 16px;
      line-height: 1.65;
    }}
    .lead {{ font-size: 18px; color: var(--muted); }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 768px) {{
      .kpis, .plots, .comparison-plots, .summary-compare, .criteria-grid {{ grid-template-columns: 1fr; }}
      section {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Informe de Imputación de Datos Faltantes</h1>
    <p>Dataset: {_escape(metadata.get("dataset_name", "dataset"))}</p>
    <p>Fecha: {_escape(metadata.get("fecha_generacion", "No aplica"))}</p>
    <div class="kpis">
      <div class="kpi"><span class="label">Filas x Columnas</span><span class="value">{_escape(metadata.get("n_rows"))} x {_escape(metadata.get("n_columns"))}</span></div>
      <div class="kpi"><span class="label">Método elegido</span><span class="value"><span class="badge {_escape(method)}">{_escape(method)}</span></span></div>
      <div class="kpi"><span class="label">Advertencias</span><span class="value"><span class="badge {'alert' if warning_count > 0 else 'ok'}">{warning_count}</span></span></div>
    </div>
  </header>
  <nav>
    <a href="#interpretacion">Interpretacion</a>
    <a href="#exploracion">Exploración</a>
    <a href="#resumen">Resumen</a>
    <a href="#decision">Decisión</a>
    <a href="#imputacion">Imputación</a>
    {'<a href="#beta-fit">Ajuste Beta</a>' if beta_section else ''}
    <a href="#supuestos">Supuestos</a>
    <a href="#advertencias">Advertencias y Sugerencias</a>
  </nav>
  <main>
    {_section_interpretation(report)}
    {_section_exploration(report, interpretation)}
    {_section_descriptive_summary(report, interpretation)}
    {_section_decision(report, interpretation)}
    {_section_imputation(report, interpretation)}
    {beta_section}
    {_section_assumptions(report, interpretation)}
    {_section_warnings(report, interpretation)}
  </main>
  <script>
    document.querySelectorAll('.var-selector-btn[data-toggle-group]').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        var group = btn.getAttribute('data-toggle-group');
        var target = btn.getAttribute('data-target');
        document.querySelectorAll('.var-selector-btn[data-toggle-group="' + group + '"]').forEach(function(item) {{ item.classList.remove('active'); }});
        document.querySelectorAll('.toggle-panel[data-toggle-group="' + group + '"]').forEach(function(item) {{ item.classList.remove('active'); }});
        btn.classList.add('active');
        var panel = document.getElementById(target);
        if (panel) {{ panel.classList.add('active'); }}
      }});
    }});
    var links = Array.from(document.querySelectorAll('nav a'));
    var observer = new IntersectionObserver(function(entries) {{
      entries.forEach(function(entry) {{
        if (entry.isIntersecting) {{
          links.forEach(function(link) {{ link.classList.remove('active'); }});
          var active = document.querySelector('nav a[href="#' + entry.target.id + '"]');
          if (active) {{ active.classList.add('active'); }}
        }}
      }});
    }}, {{ rootMargin: '-35% 0px -55% 0px', threshold: 0 }});
    document.querySelectorAll('section[id]').forEach(function(section) {{ observer.observe(section); }});
    document.querySelectorAll('.katex-formula').forEach(function(el) {{
      if (window.katex) {{
        katex.render(el.textContent, el, {{throwOnError: false}});
      }}
    }});
  </script>
</body>
</html>
"""


def _section_interpretation(report: dict[str, Any]) -> str:
    interpretation = report.get("interpretacion_ia")
    if interpretation is None:
        content = "<p>Interpretacion de IA no disponible para esta corrida.</p>"
    else:
        if isinstance(interpretation, dict):
            interpretation = interpretation.get("general")
        paragraphs = [
            paragraph.strip()
            for paragraph in str(interpretation).split("\n\n")
            if paragraph.strip()
        ]
        content = "".join(f"<p>{_escape(paragraph)}</p>" for paragraph in paragraphs)
        if not content:
            content = "<p>Interpretacion de IA no disponible para esta corrida.</p>"

    return (
        '<section id="interpretacion" class="ai-section">'
        '<h2 class="ai-title">Interpretacion del Agente (IA) '
        '<span class="ai-badge">IA</span></h2>'
        f'<div class="ai-content">{content}</div></section>'
    )


def _section_exploration(
    report: dict[str, Any],
    interpretation: Any,
) -> str:
    exploration = report.get("exploracion_previa") or {}
    missing_rows = _missing_summary_rows(exploration)
    parts = ['<section id="exploracion"><h2>Exploración Previa</h2>']
    parts.append(_scrollable_table([["Variable", "% faltantes"]] + missing_rows) if missing_rows else "<p>No aplica</p>")

    if exploration.get("numeric_analysis_skipped"):
        skipped = exploration["numeric_analysis_skipped"]
        parts.append(f'<p class="muted">Análisis numérico omitido: {_escape(skipped.get("reason"))}</p>')
    else:
        parts.append(_key_value_block("Test de Little (naniar)", exploration.get("mcar_test_naniar")))
        parts.append(_key_value_block("Test de Little (manual)", exploration.get("little_test_manual")))
        parts.append(_key_value_block("Estimaciones EM manuales", exploration.get("em_estimates_manual"), latex_labels=True))
        parts.append(_key_value_block("Estimaciones EM mvnmle", exploration.get("em_estimates_mvnmle"), latex_labels=True))

    parts.append("<h3>Gráficos de Diagnóstico</h3>")
    parts.append(_plots_grid(exploration.get("plots_generated", [])))
    parts.append(_ai_note(interpretation, "exploracion"))
    parts.append("</section>")
    return "\n".join(parts)


def _section_descriptive_summary(
    report: dict[str, Any],
    interpretation: Any,
) -> str:
    summary = report.get("resumen_descriptivo")
    comparison_plots = (report.get("exploracion_previa") or {}).get("comparison_plots", {})
    if not summary:
        return (
            '<section id="resumen"><h2>Resumen Descriptivo: Antes vs. Después</h2>'
            "<p>No aplica</p>"
            f'{_ai_note(interpretation, "resumen_descriptivo")}</section>'
        )

    buttons = []
    panels = []
    for index, variable in enumerate(summary):
        safe_id = _safe_dom_id(variable)
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button class="var-selector-btn{active}" data-toggle-group="resumen" '
            f'data-target="resumen-{safe_id}">{_escape(variable)}</button>'
        )
        panels.append(_variable_summary_panel(variable, safe_id, summary[variable], comparison_plots.get(variable), active))

    return (
        '<section id="resumen"><h2>Resumen Descriptivo: Antes vs. Después</h2>'
        f'<div class="selector">{"".join(buttons)}</div>'
        f'{"".join(panels)}{_ai_note(interpretation, "resumen_descriptivo")}</section>'
    )


def _variable_summary_panel(
    variable: str,
    safe_id: str,
    data: dict[str, Any],
    plots: dict[str, str] | None,
    active: str,
) -> str:
    antes = data.get("antes") or {}
    despues = data.get("despues") or {}
    fields = ["media", "mediana", "desviacion_estandar", "minimo", "maximo"]
    unchanged = all(antes.get(field) == despues.get(field) for field in fields)
    before_rows = "".join(f'<div class="metric-row"><span>{_escape(field)}</span><strong>{_escape(antes.get(field))}</strong></div>' for field in fields)
    after_rows = "".join(f'<div class="metric-row"><span>{_escape(field)}</span><strong>{_escape(despues.get(field))}</strong></div>' for field in fields)
    plot_html = ""
    if unchanged:
        plot_html = '<p class="muted">Esta variable no tuvo valores faltantes, no hubo cambios.</p>'
    elif plots:
        plot_html = _comparison_plots_grid(plots)

    return (
        f'<div id="resumen-{safe_id}" class="variable-summary toggle-panel{active}" data-toggle-group="resumen">'
        f"<h3>{_escape(variable)}</h3>"
        '<div class="summary-compare">'
        f'<div class="summary-card"><h3>Antes</h3>{before_rows}</div>'
        f'<div class="summary-card"><h3>Después</h3>{after_rows}</div>'
        "</div>"
        f"{plot_html}</div>"
    )


def _section_decision(
    report: dict[str, Any],
    interpretation: Any,
) -> str:
    decision = report.get("decision") or {}
    method = decision.get("metodo", "No aplica")
    criteria = decision.get("criterios") or {}
    icon = "🔀" if method == "mice" else ("📈" if method == "regresion_estocastica" else "")
    goal = criteria.get("goal")
    goal_label = {
        "inference": "Inferencia estadística",
        "prediction": "Predicción rápida",
    }.get(goal, goal)

    return (
        '<section id="decision"><h2>Decisión del Método</h2>'
        '<div class="decision-card">'
        '<div class="method-row">'
        f'<span class="badge {_escape(method)} method-badge">{icon} {_escape(method)}</span>'
        "</div>"
        '<div class="criteria-grid">'
        f'{_stat_pill("% faltantes (peor variable)", criteria.get("max_pct"))}'
        f'{_stat_pill("Ratio filas/columnas", criteria.get("ratio"))}'
        f'{_stat_pill("Objetivo del análisis", goal_label)}'
        "</div>"
        f'<p class="decision-reasoning">{_escape(decision.get("razonamiento", "No aplica"))}</p>'
        f'</div>{_ai_note(interpretation, "decision")}</section>'
    )


def _section_imputation(
    report: dict[str, Any],
    interpretation: Any,
) -> str:
    imputation = report.get("imputacion")
    if not isinstance(imputation, dict):
        content = "<p>No aplica</p>"
    else:
        point_rows = [["Variable", "Valor"]]
        point_rows.extend(_dict_rows("", imputation.get("point_estimate")))
        severity_rows = [["Campo", "Valor"]]
        severity_rows.extend(_dict_rows("", imputation.get("severity")))
        point_content = (
            _scrollable_table(point_rows) if len(point_rows) > 1 else "<p>No aplica</p>"
        )
        severity_content = (
            _scrollable_table(severity_rows, latex_first_col=True)
            if len(severity_rows) > 1
            else "<p>No aplica</p>"
        )
        content = (
            '<div class="selector">'
            '<button class="var-selector-btn active" data-toggle-group="imputacion" '
            'data-target="imputacion-point-estimate">Point Estimate</button>'
            '<button class="var-selector-btn" data-toggle-group="imputacion" '
            'data-target="imputacion-severidad">Severidad</button>'
            "</div>"
            '<div id="imputacion-point-estimate" class="toggle-panel active" data-toggle-group="imputacion">'
            f"{point_content}</div>"
            '<div id="imputacion-severidad" class="toggle-panel" data-toggle-group="imputacion">'
            f"{severity_content}</div>"
        )
    return (
        '<section id="imputacion"><h2>Resultado de la Imputación</h2>'
        f'{content}{_ai_note(interpretation, "imputacion")}</section>'
    )


def _section_beta_distribution_fit(report: dict[str, Any]) -> str:
    imputation = report.get("imputacion")
    beta_fit = (
        imputation.get("beta_distribution_fit")
        if isinstance(imputation, dict)
        else None
    )
    if not isinstance(beta_fit, dict) or not beta_fit:
        return ""

    buttons = []
    panels = []
    for index, variable in enumerate(beta_fit):
        safe_id = _safe_dom_id(variable)
        active = " active" if index == 0 else ""
        buttons.append(
            f'<button class="var-selector-btn{active}" data-toggle-group="beta-fit" '
            f'data-target="beta-fit-{safe_id}">{_escape(variable)}</button>'
        )
        panels.append(_beta_fit_panel(variable, safe_id, beta_fit[variable], active))

    return (
        '<section id="beta-fit"><h2>Ajuste de Distribución Beta</h2>'
        f'<div class="selector">{"".join(buttons)}</div>'
        f'{"".join(panels)}</section>'
    )


def _beta_fit_panel(
    variable: str,
    safe_id: str,
    payload: dict[str, Any],
    active: str,
) -> str:
    alpha = payload.get("shape1_pooled")
    beta = payload.get("shape2_pooled")
    scale_used = payload.get("scale_used")
    n_imputations = payload.get("n_imputations")
    return (
        f'<div id="beta-fit-{safe_id}" class="toggle-panel{active}" data-toggle-group="beta-fit">'
        f"<h3>{_escape(variable)}</h3>"
        '<div class="summary-compare">'
        '<div class="summary-card">'
        '<h3>Parámetros pooled</h3>'
        f'<div class="metric-row"><span class="katex-formula">\\alpha = {_escape(alpha)}</span></div>'
        f'<div class="metric-row"><span class="katex-formula">\\beta = {_escape(beta)}</span></div>'
        "</div>"
        '<div class="summary-card">'
        "<h3>Escala</h3>"
        f'<div class="metric-row"><span>Escala usada</span><strong>{_escape(scale_used)}</strong></div>'
        f'<div class="metric-row"><span>Imputaciones</span><strong>{_escape(n_imputations)}</strong></div>'
        "</div>"
        "</div>"
        f"{_beta_severity_table(payload)}</div>"
    )


def _beta_severity_table(payload: dict[str, Any]) -> str:
    per_parameter = payload.get("severity_per_parameter") or {}
    joint = payload.get("severity_joint") or {}
    rows = [["Métrica", "shape1", "shape2", "conjunta"]]

    for metric in ["lambda", "r", "df", "gamma"]:
        shape1 = per_parameter.get("shape1", {}).get(metric)
        shape2 = per_parameter.get("shape2", {}).get(metric)
        rows.append([metric, _text(shape1), _text(shape2), _text(joint.get(metric))])

    return _scrollable_table(rows, latex_first_col=True)


def _section_assumptions(
    report: dict[str, Any],
    interpretation: Any,
) -> str:
    imputation = report.get("imputacion") or {}
    assumptions = imputation.get("assumptions") if isinstance(imputation, dict) else None
    if not assumptions:
        content = "<p>No aplica</p>"
    else:
        rows, failing_indexes = _assumption_rows(assumptions)
        content = _scrollable_table(rows, failing_indexes) if len(rows) > 1 else "<p>No aplica</p>"
    return (
        '<section id="supuestos"><h2>Cumplimiento de Supuestos</h2>'
        f'{content}{_ai_note(interpretation, "supuestos")}</section>'
    )


def _section_warnings(
    report: dict[str, Any],
    interpretation: Any,
) -> str:
    warnings = report.get("advertencias") or []
    if not warnings:
        content = "<p>Sin advertencias</p>"
    else:
        content = "<ul>" + "".join(f"<li>{_escape(warning)}</li>" for warning in warnings) + "</ul>"
    return (
        '<section id="advertencias"><h2>Advertencias y Sugerencias</h2>'
        f'{content}{_ai_note(interpretation, "advertencias_y_sugerencias")}</section>'
    )


def _ai_note(interpretation: Any, key: str) -> str:
    if not isinstance(interpretation, dict):
        return ""
    value = interpretation.get(key)
    if value is None or not str(value).strip():
        return ""
    if key == "advertencias_y_sugerencias" and isinstance(value, dict):
        resumen = value.get("resumen")
        acciones = value.get("acciones_sugeridas")
        parts = []
        if resumen is not None and str(resumen).strip():
            parts.append(f"<p>{_escape(resumen)}</p>")
        if isinstance(acciones, list) and acciones:
            items = "".join(
                f"<li>{_escape(action)}</li>"
                for action in acciones
                if str(action).strip()
            )
            if items:
                parts.append(f"<ul>{items}</ul>")
        if not parts:
            return ""
        return (
            '<div class="ai-note"><strong>Interpretacion:</strong> '
            f"{''.join(parts)}</div>"
        )
    return (
        '<div class="ai-note"><strong>Interpretacion:</strong> '
        f'{_escape(value)}</div>'
    )


def _stat_pill(label: str, value: Any) -> str:
    return (
        '<div class="stat-pill">'
        f'<span class="label">{_escape(label)}</span>'
        f'<span class="value">{_escape(value)}</span>'
        "</div>"
    )


def _plots_grid(plots: list[str]) -> str:
    cards = []
    for plot in plots:
        plot_path = Path(plot)
        if not plot_path.exists():
            continue
        encoded = base64.b64encode(plot_path.read_bytes()).decode("ascii")
        name = plot_path.name
        title = FRIENDLY_PLOT_NAMES.get(name, name)
        cards.append(
            '<div class="plot-card">'
            f"<h3>{_escape(title)}</h3>"
            f'<img src="data:image/png;base64,{encoded}" alt="{_escape(title)}">'
            "</div>"
        )
    if not cards:
        return "<p>No aplica</p>"
    return '<div class="plots">' + "".join(cards) + "</div>"


def _comparison_plots_grid(plots: dict[str, str]) -> str:
    labels = {
        "boxplot": "Boxplot antes vs. después",
        "histograma": "Histograma antes vs. después",
        "qqplot": "QQ-plot",
    }
    cards = []
    for kind in ["boxplot", "histograma", "qqplot"]:
        plot_html = plots.get(kind)
        if not plot_html:
            continue
        title = labels[kind]
        cards.append(
            '<div class="plot-card">'
            f"<h3>{_escape(title)}</h3>"
            f"{plot_html}"
            "</div>"
        )
    if not cards:
        return ""
    return '<div class="comparison-plots">' + "".join(cards) + "</div>"


def _missing_summary_rows(exploration: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    summaries = exploration.get("md_pattern_summary", {}).get("miss_var_summary", [])
    if isinstance(summaries, list):
        for entry in summaries:
            if not isinstance(entry, dict):
                continue
            variable = entry.get("variable") or entry.get("var") or entry.get("feature")
            pct = entry.get("pct_miss") or entry.get("pct_missing") or entry.get("missing_percent")
            if variable is not None and pct is not None:
                rows.append([_text(variable), _text(pct)])
    return rows


def _key_value_block(title: str, payload: Any, latex_labels: bool = False) -> str:
    if not payload:
        return f"<h3>{_escape(title)}</h3><p>No aplica</p>"
    if isinstance(payload, dict):
        rows = [["Campo", "Valor"]] + _dict_rows("", payload)
        return f"<h3>{_escape(title)}</h3>{_scrollable_table(rows, latex_first_col=latex_labels)}"
    return f"<h3>{_escape(title)}</h3><p>{_escape(payload)}</p>"


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
            rows.append([
                _text(variable),
                "convergence",
                "Convergencia de cadenas MICE",
                "Convergio" if converged_value is True else ("No convergio" if converged_value is False else "No aplica"),
                _text(converged_value),
            ])
            if converged_value is False:
                failing_indexes.append(len(rows) - 1)

    distribution = assumptions.get("distribution_comparison", {})
    if isinstance(distribution, dict):
        for variable, payload in distribution.items():
            _append_test_row(rows, failing_indexes, variable, "distribution_comparison", payload, "Anderson-Darling (k-muestras)")

    for variable, payload in assumptions.items():
        if variable in ("convergence", "distribution_comparison") or not isinstance(payload, dict):
            continue
        _append_test_row(rows, failing_indexes, variable, "normality", payload.get("normality"))
        _append_test_row(rows, failing_indexes, variable, "homoscedasticity", payload.get("homoscedasticity"))

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
    rows.append([
        _text(variable),
        assumption_name,
        _text(payload.get("test_used", default_test_label)),
        f"stat={_text(payload.get('statistic', payload.get('ad_statistic')))}; p={_text(payload.get('p_value'))}",
        _text(payload.get("meets_assumption")),
    ])
    if payload.get("meets_assumption") is False:
        failing_indexes.append(len(rows) - 1)


def _scrollable_table(
    rows: list[list[str]],
    failing_indexes: list[int] | None = None,
    max_visible_rows: int = 10,
    latex_first_col: bool = False,
) -> str:
    table = _html_table(rows, failing_indexes, latex_first_col=latex_first_col)
    if len(rows) <= max_visible_rows:
        return table
    return f'<div class="table-scroll">{table}</div>'


def _html_table(
    rows: list[list[str]],
    failing_indexes: list[int] | None = None,
    latex_first_col: bool = False,
) -> str:
    failing = set(failing_indexes or [])
    html_rows = []
    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        class_attr = ' class="fila-alerta"' if index in failing else ""
        cells = []
        for col_index, cell in enumerate(row):
            if latex_first_col and index > 0 and col_index == 0:
                content = f'<span class="katex-formula">{_escape(_to_latex_label(str(cell)))}</span>'
            else:
                content = _escape(cell)
            cells.append(f"<{tag}>{content}</{tag}>")
        html_rows.append(f"<tr{class_attr}>{''.join(cells)}</tr>")
    return "<table><thead>" + html_rows[0] + "</thead><tbody>" + "".join(html_rows[1:]) + "</tbody></table>"


def _to_latex_label(key: str) -> str:
    simple_labels = {
        "lambda": "\\lambda",
        "r": "r",
        "df": "df",
        "gamma": "\\gamma",
    }
    if key in simple_labels:
        return simple_labels[key]

    parts = key.split(".")
    if len(parts) > 1 and parts[0] in simple_labels:
        suffix = ",".join(parts[1:])
        return f"{simple_labels[parts[0]]}_{{{suffix}}}"
    if len(parts) == 2 and parts[0] == "mu":
        return f"\\mu_{{{parts[1]}}}"
    if len(parts) == 3 and parts[0] == "sigma":
        return f"\\sigma_{{{parts[1]},{parts[2]}}}"
    return key


def _safe_dom_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return safe or "variable"


def _text(value: Any) -> str:
    if value is None:
        return "No aplica"
    if isinstance(value, float):
        rounded = round(value, 4)
        return f"{rounded:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(_text(value), quote=True)
