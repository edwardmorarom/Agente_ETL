from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


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
from scripts.generate_html_report import generate_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline ETL completo.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--goal",
        choices=["inference", "prediction"],
        default="prediction",
    )
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--vars", default=None)
    parser.add_argument("--beta_vars", default=None)
    parser.add_argument("--skip_ai", action="store_true", default=False)
    parser.add_argument("--domain_context", default=None)
    args = parser.parse_args(argv)

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

    clean_csv = output_dir / "datos_limpios.csv"
    profile_json = output_dir / "perfil.json"
    df.to_csv(clean_csv, index=False)
    with open(profile_json, "w", encoding="utf-8") as profile_file:
        json.dump(profile, profile_file, indent=2, ensure_ascii=False, default=str)

    diagnostico = DiagnosticoRunner(plots_dir=plots_dir).run(df)
    mice_vars = _parse_vars(args.vars)
    beta_vars = _parse_vars(args.beta_vars)
    missing_vars = [col for col in df.columns if df[col].isna().sum() > 0]
    pipeline_result = ImputationPipeline(
        mice_vars=mice_vars,
        beta_vars=beta_vars,
    ).run(df, goal=args.goal)

    imputed_data = pipeline_result["imputed_data"]
    imputed_csv = output_dir / "datos_imputados.csv"
    imputed_data.to_csv(imputed_csv, index=False)
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

    explainer = None
    if not args.skip_ai:
        from llm.client import LLMRequestError
        from llm.explainer import PipelineExplainer

        try:
            explainer = PipelineExplainer(
                report,
                domain_context=args.domain_context,
            )
            interpretacion = explainer.explain()
            report["interpretacion_ia"] = interpretacion
            with open(report_json_path, "w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2, ensure_ascii=False, default=str)
        except (LLMRequestError, ValueError) as exc:
            print(
                "No se pudo generar la interpretacion de IA: "
                f"{exc}. Continuando sin ella."
            )
            explainer = None

    generate_html(report_json_path, output_dir / "reporte.html")

    print(f"Listo. Resultados en: {output_dir}")
    print(f"Abre {output_dir / 'reporte.html'} para ver el dashboard.")

    if explainer is not None:
        from llm.client import LLMRequestError

        print("\n¿Quieres hacer preguntas sobre este resultado? (s/n): ", end="")
        respuesta = input().strip().lower()
        if respuesta == "s":
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


if __name__ == "__main__":
    raise SystemExit(main())
