from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.comparison_plots import build_comparison_plots
from core.ingestion import (
    build_profile,
    detect_id_columns,
    drop_rows_without_information,
    load_file,
    standardize,
)
from core.pipeline import ImputationPipeline
from core.report_builder import build_report_json
from imputers.diagnostico_runner import DiagnosticoRunner
from imputers.mice_imputer import MiceImputer
from scripts.generate_html_report import generate_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline ETL completo.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--goal",
        choices=["inference", "prediction"],
        default=None,
    )
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--vars", default=None)
    parser.add_argument("--beta_vars", default=None)
    parser.add_argument("--skip_ai", action="store_true", default=False)
    parser.add_argument("--no_auto_retry", action="store_true", default=False)
    parser.add_argument("--no_interactive", action="store_true", default=False)
    parser.add_argument("--domain_context", default=None)
    args = parser.parse_args(argv)

    if not args.no_interactive:
        print("=== Agente de Imputacion de Datos Faltantes ===")
        print(
            "Voy a hacerte algunas preguntas rapidas antes de empezar. "
            "Si ya sabes que parametros quieres usar, puedes pasarlos "
            "como argumentos y me saltare esa pregunta.\n"
        )

    dataset_name = args.dataset_name or args.input.stem
    output_dir = _make_output_dir(dataset_name)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_file(args.input)
    profile = build_profile(df_raw)
    df = standardize(df_raw, profile)
    id_columns = detect_id_columns(df)
    df, dropped_rows = drop_rows_without_information(df, id_columns)
    profile["rows_dropped_no_information"] = {
        "threshold_pct": 80.0,
        "id_columns_detected": id_columns,
        "count": len(dropped_rows),
        "dropped_rows": dropped_rows,
    }

    if args.domain_context is None and not args.no_interactive:
        print(
            "\n¿De que se tratan estos datos? (por ejemplo: "
            "'encuesta de ingresos de hogares en Bogota', 'datos "
            "de un experimento de laboratorio'). Si no lo sabes o "
            "prefieres omitirlo, presiona Enter."
        )
        respuesta = input("> ").strip()
        if respuesta:
            args.domain_context = respuesta

    if args.goal is None:
        if not args.no_interactive:
            print("\n¿Que necesitas hacer con estos datos?")
            print(
                "  1. Sacar conclusiones estadisticas confiables "
                "(con margen de error), por ejemplo para un informe "
                "o una investigacion."
            )
            print(
                "  2. Solo rellenar los datos faltantes rapido, "
                "para seguir trabajando con el archivo completo."
            )
            eleccion = input("Escribe 1 o 2 (Enter = opcion 2): ").strip()
            args.goal = "inference" if eleccion == "1" else "prediction"
        else:
            args.goal = "prediction"

    diagnostico = DiagnosticoRunner(plots_dir=plots_dir).run(df)
    mice_vars = _parse_vars(args.vars)
    beta_vars = _parse_vars(args.beta_vars)
    pipeline_result = ImputationPipeline(
        mice_vars=mice_vars,
        beta_vars=beta_vars,
    ).run(df, goal=args.goal)
    pipeline_result = _auto_retry_mice_if_needed(
        pipeline_result,
        df,
        args.goal,
        mice_vars,
        beta_vars,
        disabled=args.no_auto_retry,
    )

    report, report_json_path = _write_pipeline_outputs(
        output_dir,
        df,
        profile,
        diagnostico,
        pipeline_result,
        dataset_name,
        render_html=False,
    )

    explainer = None
    if not args.skip_ai:
        from llm.client import LLMRequestError
        from llm.explainer import PipelineExplainer

        def ejecutar_recalculo_completo(parametros: dict) -> tuple[Any, dict[str, Any]]:
            return _run_recalculation(
                parametros,
                df,
                args.goal,
                mice_vars,
                beta_vars,
            )

        def ejecutar_recalculo(parametros: dict) -> dict[str, Any]:
            _recalc_df, recalc_result = ejecutar_recalculo_completo(parametros)
            return _compact_pipeline_result(recalc_result)

        try:
            explainer = PipelineExplainer(
                report,
                domain_context=args.domain_context,
                tool_executor=ejecutar_recalculo,
            )
            interpretacion = explainer.explain()
            report["interpretacion_ia"] = interpretacion
        except (LLMRequestError, ValueError) as exc:
            print(
                "No se pudo generar la interpretacion de IA: "
                f"{exc}. Continuando sin ella."
            )
            explainer = None

    _write_report_json_and_html(report, report_json_path, output_dir)

    if explainer is not None and not args.no_interactive:
        _offer_and_apply_suggestion(
            report,
            explainer,
            ejecutar_recalculo_completo,
            dataset_name,
        )

    print(f"Listo. Resultados en: {output_dir}")
    print(f"Abre {output_dir / 'reporte.html'} para ver el dashboard.")

    if explainer is not None and not args.no_interactive:
        from llm.client import LLMRequestError

        print("\n¿Quieres hacer preguntas sobre este resultado? (s/n): ", end="")
        respuesta = input().strip().lower()
        if respuesta == "s":
            print(
                "Tip: puedes pedirme que recalcule con otros parametros, "
                "por ejemplo 'prueba con mas imputaciones' o 'cambia el "
                "objetivo a prediccion'."
            )
            print("Escribe tus preguntas. Escribe 'salir' para terminar.\n")
            while True:
                pregunta = input("Tu pregunta: ").strip()
                if pregunta.lower() in ("salir", "exit"):
                    break
                if not pregunta:
                    continue
                try:
                    respuesta_ia = explainer.ask(pregunta)
                    print(f"\nAgente: {respuesta_ia}\n")
                except (LLMRequestError, ValueError) as exc:
                    print(f"\nError al consultar la IA: {exc}\n")

    return 0


def _make_output_dir(dataset_name: str) -> Path:
    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in dataset_name.strip()
    ).strip("_") or "dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"{safe_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _parse_vars(raw_vars: str | None) -> list[str] | None:
    if raw_vars is None:
        return None
    return [value.strip() for value in raw_vars.split(",") if value.strip()]


def _write_pipeline_outputs(
    output_dir: Path,
    df: pd.DataFrame,
    profile: dict[str, Any],
    diagnostico: dict[str, Any],
    pipeline_result: dict[str, Any],
    dataset_name: str,
    render_html: bool = True,
) -> tuple[dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "datos_limpios.csv", index=False)
    with open(output_dir / "perfil.json", "w", encoding="utf-8") as profile_file:
        json.dump(profile, profile_file, indent=2, ensure_ascii=False, default=str)

    imputed_data = pipeline_result["imputed_data"]
    imputed_data.to_csv(output_dir / "datos_imputados.csv", index=False)
    missing_vars = [col for col in df.columns if df[col].isna().sum() > 0]
    comparison_plots = build_comparison_plots(df, imputed_data, missing_vars)

    report_json_path = output_dir / "reporte.json"
    report = build_report_json(
        diagnostico,
        pipeline_result,
        dataset_name=dataset_name,
        output_path=report_json_path,
        df_original=df,
        comparison_plots=comparison_plots,
    )

    if render_html:
        _write_report_json_and_html(report, report_json_path, output_dir)

    return report, report_json_path


def _write_report_json_and_html(
    report: dict[str, Any],
    report_json_path: Path,
    output_dir: Path,
) -> None:
    with open(report_json_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, default=str)
    generate_html(report_json_path, output_dir / "reporte.html")


def _run_recalculation(
    parametros: dict[str, Any],
    df: pd.DataFrame,
    original_goal: str,
    mice_vars: list[str] | None,
    beta_vars: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recalc_df = _df_for_recalculation(df, parametros)
    excluded = _parse_excluded_variables(parametros)
    recalc_goal = parametros.get("goal", original_goal)
    recalc_mice_vars = _filter_existing_vars(mice_vars, recalc_df)
    requested_beta_vars = parametros.get("beta_vars", beta_vars)
    recalc_beta_vars = _filter_existing_vars(requested_beta_vars, recalc_df)

    if "m" in parametros and parametros["m"] is not None:
        m = int(parametros["m"])
        imputer = MiceImputer(vars=recalc_mice_vars, beta_vars=recalc_beta_vars, m=m)
        imputed_data = imputer.fit_transform(recalc_df)
        max_pct = float((recalc_df.isna().mean() * 100).max()) if recalc_df.shape[1] else 0.0
        ratio = len(recalc_df) / recalc_df.shape[1] if recalc_df.shape[1] else float("inf")
        pipeline = ImputationPipeline()
        result = {
            "decision": "mice",
            "reasoning": f"Se recalculo la imputacion MICE con m={m} por solicitud del usuario.",
            "criteria": {
                "max_pct": round(max_pct, 2),
                "ratio": round(ratio, 2),
                "goal": recalc_goal,
            },
            "imputed_data": imputed_data,
            "imputer_report": imputer.last_report,
            "warnings": pipeline._build_lambda_warnings(imputer.last_report),
        }
    else:
        result = ImputationPipeline(
            mice_vars=recalc_mice_vars,
            beta_vars=recalc_beta_vars,
        ).run(recalc_df, goal=recalc_goal)

    if excluded:
        result["reasoning"] = (
            f"{result.get('reasoning', '')} Se excluyeron del recalculo las "
            f"variables: {', '.join(excluded)}."
        ).strip()

    return recalc_df, result


def _df_for_recalculation(
    df: pd.DataFrame,
    parametros: dict[str, Any],
) -> pd.DataFrame:
    excluded = _parse_excluded_variables(parametros)
    if not excluded:
        return df.copy()

    existing_columns = [column for column in excluded if column in df.columns]
    return df.drop(columns=existing_columns)


def _parse_excluded_variables(parametros: dict[str, Any]) -> list[str]:
    excluded = parametros.get("excluir_variables")
    if not isinstance(excluded, list):
        return []
    return [str(column).strip() for column in excluded if str(column).strip()]


def _filter_existing_vars(
    variables: Any,
    df: pd.DataFrame,
) -> list[str] | None:
    if variables is None:
        return None
    if not isinstance(variables, list):
        return None

    filtered = [
        str(variable).strip()
        for variable in variables
        if str(variable).strip() in df.columns
    ]
    return filtered or None


def _profile_for_output(df: pd.DataFrame) -> dict[str, Any]:
    profile = build_profile(df)
    id_columns = detect_id_columns(df)
    profile["rows_dropped_no_information"] = {
        "threshold_pct": 80.0,
        "id_columns_detected": id_columns,
        "count": 0,
        "dropped_rows": [],
    }
    return profile


def _extract_suggested_actions(report: dict[str, Any]) -> list[str]:
    interpretation = report.get("interpretacion_ia")
    if not isinstance(interpretation, dict):
        return []

    suggestions_payload = interpretation.get("advertencias_y_sugerencias")
    if not isinstance(suggestions_payload, dict):
        return []

    suggestions = suggestions_payload.get("acciones_sugeridas", [])
    if not isinstance(suggestions, list):
        return []

    return [str(suggestion) for suggestion in suggestions if str(suggestion).strip()]


def _offer_and_apply_suggestion(
    report: dict[str, Any],
    explainer: Any,
    ejecutar_recalculo_completo: Any,
    dataset_name: str,
) -> None:
    sugerencias = _extract_suggested_actions(report)
    if not sugerencias:
        return

    print("\nEl agente encontro posibles formas de mejorar este resultado:")
    for index, sugerencia in enumerate(sugerencias, 1):
        print(f"  {index}. {sugerencia}")

    eleccion = input(
        "\nQuieres que aplique alguna? Escribe el numero, o Enter para omitir: "
    ).strip()
    if not eleccion.isdigit() or not 1 <= int(eleccion) <= len(sugerencias):
        return

    sugerencia_elegida = sugerencias[int(eleccion) - 1]
    prompt_traduccion = (
        f"Convierte esta sugerencia a JSON de accion: "
        f'"{sugerencia_elegida}". Responde UNICAMENTE con: '
        '{"accion": "recalcular", "parametros": {"m": '
        'numero_opcional, "goal": "inference_o_prediction_opcional", '
        '"beta_vars": ["lista_opcional"], "excluir_variables": '
        '["lista_opcional"]}} - incluye solo los campos que '
        "la sugerencia pide cambiar."
    )
    respuesta_llm = explainer.client.chat(
        [{"role": "user", "content": prompt_traduccion}],
        generation_options={"temperature": 0.0, "num_predict": 200},
    )

    from llm.explainer import _try_parse_action

    accion = _try_parse_action(respuesta_llm)
    if accion is None:
        print(
            "No pude traducir esa sugerencia a una accion ejecutable. Puedes "
            "intentarlo manualmente en el modo de preguntas."
        )
        return

    print("\nAplicando la sugerencia y generando un nuevo resultado...\n")
    recalc_df, nuevo_pipeline_result = ejecutar_recalculo_completo(accion["parametros"])
    nueva_carpeta = _make_output_dir(f"{dataset_name}_mejorado")
    new_plots_dir = nueva_carpeta / "plots"
    new_plots_dir.mkdir(parents=True, exist_ok=True)
    new_profile = _profile_for_output(recalc_df)
    new_diagnostico = DiagnosticoRunner(plots_dir=new_plots_dir).run(recalc_df)
    _write_pipeline_outputs(
        nueva_carpeta,
        recalc_df,
        new_profile,
        new_diagnostico,
        nuevo_pipeline_result,
        f"{dataset_name}_mejorado",
        render_html=True,
    )
    print(f"Nuevo resultado en: {nueva_carpeta}")


def _auto_retry_mice_if_needed(
    pipeline_result: dict[str, Any],
    df,
    goal: str,
    mice_vars: list[str] | None,
    beta_vars: list[str] | None,
    disabled: bool = False,
) -> dict[str, Any]:
    if disabled:
        return pipeline_result
    if pipeline_result.get("decision") != "mice":
        return pipeline_result
    original_warnings = pipeline_result.get("warnings") or []
    if not original_warnings:
        return pipeline_result

    print(
        "Se detecto severidad alta (lambda). Reintentando con mas "
        "imputaciones para mejorar la confiabilidad..."
    )
    imputer_report = pipeline_result.get("imputer_report") or {}
    m_actual = imputer_report.get("n_imputations", 5)
    nuevo_m = min(int(m_actual) * 2, 20)

    retry_imputer = MiceImputer(
        vars=mice_vars,
        beta_vars=beta_vars,
        m=nuevo_m,
    )
    retry_imputed_data = retry_imputer.fit_transform(df)
    retry_warnings = ImputationPipeline()._build_lambda_warnings(
        retry_imputer.last_report
    )

    if len(retry_warnings) < len(original_warnings):
        print(
            f"Reintento exitoso con m={nuevo_m}: la severidad reporto menos "
            "advertencias."
        )
        retry_result = dict(pipeline_result)
        retry_result["imputed_data"] = retry_imputed_data
        retry_result["imputer_report"] = retry_imputer.last_report
        retry_result["warnings"] = retry_warnings
        retry_result["reasoning"] = (
            f"{pipeline_result.get('reasoning', '')} Se reintento "
            f"automaticamente con m={nuevo_m} y se uso este resultado porque "
            "redujo las advertencias de severidad."
        ).strip()
        return retry_result

    print(
        f"Reintento con m={nuevo_m} no redujo la severidad. Se conserva el "
        "resultado original."
    )
    pipeline_result["warnings"] = list(original_warnings)
    pipeline_result["warnings"].append(
        f"Se reintento automaticamente con m={nuevo_m} pero la severidad se "
        "mantuvo alta. Se recomienda revisar los datos o considerar un metodo "
        "distinto."
    )
    return pipeline_result


def _compact_pipeline_result(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    imputer_report = pipeline_result.get("imputer_report")
    compact_report = None
    if isinstance(imputer_report, dict):
        compact_report = {
            key: imputer_report[key]
            for key in ("severity", "point_estimate", "beta_distribution_fit")
            if key in imputer_report
        }

    return {
        "decision": pipeline_result.get("decision"),
        "reasoning": pipeline_result.get("reasoning"),
        "warnings": pipeline_result.get("warnings", []),
        "imputer_report": compact_report,
    }


if __name__ == "__main__":
    raise SystemExit(main())
