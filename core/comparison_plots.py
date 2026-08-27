from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def build_comparison_plots(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    vars_with_missing: list[str],
) -> dict[str, dict[str, str]]:
    plots: dict[str, dict[str, str]] = {}
    if not vars_with_missing:
        return plots

    for variable in vars_with_missing:
        if variable not in df_before.columns or variable not in df_after.columns:
            continue

        before_values = pd.to_numeric(df_before[variable], errors="coerce").dropna()
        after_values = pd.to_numeric(df_after[variable], errors="coerce").dropna()
        if before_values.empty or after_values.empty:
            continue

        plots[variable] = {
            "boxplot": _to_embedded_html(
                _build_boxplot(variable, before_values, after_values)
            ),
            "histograma": _to_embedded_html(
                _build_histogram(variable, before_values, after_values)
            ),
            "qqplot": _to_embedded_html(
                _build_qqplot(variable, before_values, after_values)
            ),
        }

    return plots


def _build_boxplot(
    variable: str,
    before_values: pd.Series,
    after_values: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=before_values, name="Antes"))
    fig.add_trace(go.Box(y=after_values, name="Despues"))
    fig.update_layout(
        title=f"Boxplot antes vs. despues: {variable}",
        yaxis_title=variable,
        template="plotly_white",
        height=350,
    )
    return fig


def _build_histogram(
    variable: str,
    before_values: pd.Series,
    after_values: pd.Series,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=before_values, name="Antes", opacity=0.6))
    fig.add_trace(go.Histogram(x=after_values, name="Despues", opacity=0.6))
    fig.update_layout(
        title=f"Histograma antes vs. despues: {variable}",
        xaxis_title=variable,
        yaxis_title="Frecuencia",
        barmode="overlay",
        template="plotly_white",
        height=350,
    )
    return fig


def _build_qqplot(
    variable: str,
    before_values: pd.Series,
    after_values: pd.Series,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Antes", "Despues"))
    _add_qq_trace(fig, before_values, "Antes", row=1, col=1)
    _add_qq_trace(fig, after_values, "Despues", row=1, col=2)
    fig.update_layout(
        title=f"QQ-plot: {variable}",
        template="plotly_white",
        height=350,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Cuantiles teoricos", row=1, col=1)
    fig.update_xaxes(title_text="Cuantiles teoricos", row=1, col=2)
    fig.update_yaxes(title_text="Cuantiles observados", row=1, col=1)
    fig.update_yaxes(title_text="Cuantiles observados", row=1, col=2)
    return fig


def _add_qq_trace(
    fig: go.Figure,
    values: pd.Series,
    name: str,
    row: int,
    col: int,
) -> None:
    osm, osr = stats.probplot(values, dist="norm")[0]
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode="markers", name=name),
        row=row,
        col=col,
    )


def _to_embedded_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False)
