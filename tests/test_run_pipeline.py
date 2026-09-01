from __future__ import annotations

import builtins
import json
from pathlib import Path
import shutil
import sys
import types

import pandas as pd
import pytest

import scripts.run_pipeline as run_pipeline_module
from scripts.run_pipeline import main


@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript no esta disponible en el sistema.",
)
def test_run_pipeline_creates_expected_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    rows = []
    for index in range(80):
        rows.append(
            {
                "id": f"id_{index}",
                "x": None if index == 0 else float(index),
                "y": None if index == 1 else float(index * 2),
                "z": float(index * 3 + 1),
            }
        )
    pd.DataFrame(rows).to_csv(input_csv, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *args, **kwargs: pytest.fail("input no debe llamarse con --skip_ai"),
    )

    class ForbiddenExplainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pytest.fail("PipelineExplainer no debe llamarse con --skip_ai")

    monkeypatch.setitem(
        sys.modules,
        "llm.explainer",
        types.SimpleNamespace(PipelineExplainer=ForbiddenExplainer),
    )

    exit_code = main(
        [
            "--input",
            str(input_csv),
            "--goal",
            "inference",
            "--dataset_name",
            "demo",
            "--beta_vars",
            "x",
            "--skip_ai",
            "--no_interactive",
        ]
    )

    output_dirs = list((tmp_path / "outputs").glob("demo_*"))
    assert exit_code == 0
    assert len(output_dirs) == 1
    assert (output_dirs[0] / "reporte.html").exists()
    assert (output_dirs[0] / "reporte.json").exists()
    assert (output_dirs[0] / "datos_imputados.csv").exists()

    with open(output_dirs[0] / "reporte.json", "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    assert "beta_distribution_fit" in report["imputacion"]
    assert "x" in report["imputacion"]["beta_distribution_fit"]
    assert "interpretacion_ia" not in report


def test_auto_retry_uses_improved_mice_result_and_flag_disables_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [2.0, 4.0, 6.0]})
    captured_results: list[dict[str, object]] = []

    class FakeDiagnosticoRunner:
        def __init__(self, plots_dir: Path) -> None:
            self.plots_dir = plots_dir

        def run(self, df: pd.DataFrame) -> dict[str, object]:
            return {"n_rows": len(df), "n_columns": len(df.columns)}

    class FakeImputationPipeline:
        calls = 0

        def __init__(
            self,
            mice_vars: list[str] | None = None,
            beta_vars: list[str] | None = None,
        ) -> None:
            self.mice_vars = mice_vars
            self.beta_vars = beta_vars

        def run(self, df: pd.DataFrame, goal: str) -> dict[str, object]:
            FakeImputationPipeline.calls += 1
            return {
                "decision": "mice",
                "reasoning": "resultado original",
                "criteria": {"max_pct": 33.33, "ratio": 1.5, "goal": goal},
                "imputed_data": df.fillna(9),
                "imputer_report": {
                    "n_imputations": 5,
                    "severity": {"lambda": 0.5},
                    "point_estimate": {"x": 9.0},
                },
                "warnings": ["Lambda de severidad supero el umbral 0.30 para: global."],
            }

        def _build_lambda_warnings(
            self,
            report: dict[str, object] | None,
        ) -> list[str]:
            severity = (report or {}).get("severity")
            if severity == {"lambda": 0.1}:
                return []
            return ["Lambda de severidad supero el umbral 0.30 para: global."]

    class FakeMiceImputer:
        created: list["FakeMiceImputer"] = []

        def __init__(
            self,
            vars: list[str] | None = None,
            beta_vars: list[str] | None = None,
            m: int = 5,
        ) -> None:
            self.vars = vars
            self.beta_vars = beta_vars
            self.m = m
            self.last_report = {
                "n_imputations": m,
                "severity": {"lambda": 0.1},
                "point_estimate": {"x": 1.0},
            }
            self.created.append(self)

        def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.fillna(1)

    def fake_build_report_json(
        diagnostico: dict[str, object],
        pipeline_result: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        captured_results.append(pipeline_result)
        return {
            "metadata": {"dataset_name": "demo", "n_rows": 3, "n_columns": 2},
            "exploracion_previa": diagnostico,
            "decision": {"metodo": pipeline_result["decision"]},
            "imputacion": pipeline_result["imputer_report"],
            "advertencias": pipeline_result["warnings"],
        }

    output_counter = {"value": 0}

    def fake_make_output_dir(dataset_name: str) -> Path:
        output_counter["value"] += 1
        output_dir = tmp_path / "outputs" / f"{dataset_name}_{output_counter['value']}"
        output_dir.mkdir(parents=True)
        return output_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pipeline_module, "_make_output_dir", fake_make_output_dir)
    monkeypatch.setattr(run_pipeline_module, "load_file", lambda path: df.copy())
    monkeypatch.setattr(
        run_pipeline_module,
        "build_profile",
        lambda df: {"n_rows": 3, "n_columns": 2},
    )
    monkeypatch.setattr(run_pipeline_module, "standardize", lambda df, profile: df)
    monkeypatch.setattr(run_pipeline_module, "detect_id_columns", lambda df: [])
    monkeypatch.setattr(
        run_pipeline_module,
        "drop_rows_without_information",
        lambda df, id_columns: (df, []),
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "DiagnosticoRunner",
        FakeDiagnosticoRunner,
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "ImputationPipeline",
        FakeImputationPipeline,
    )
    monkeypatch.setattr(run_pipeline_module, "MiceImputer", FakeMiceImputer)
    monkeypatch.setattr(
        run_pipeline_module,
        "build_comparison_plots",
        lambda before, after, missing_vars: {},
    )
    monkeypatch.setattr(run_pipeline_module, "build_report_json", fake_build_report_json)
    monkeypatch.setattr(
        run_pipeline_module,
        "generate_html",
        lambda report_json_path, output_html_path: None,
    )

    assert main(
        [
            "--input",
            "input.csv",
            "--goal",
            "inference",
            "--dataset_name",
            "demo",
            "--skip_ai",
            "--no_interactive",
        ]
    ) == 0
    assert FakeMiceImputer.created[0].m == 10
    assert captured_results[-1]["warnings"] == []
    assert captured_results[-1]["imputer_report"] == FakeMiceImputer.created[0].last_report

    FakeMiceImputer.created = []
    captured_results.clear()
    assert main(
        [
            "--input",
            "input.csv",
            "--goal",
            "inference",
            "--dataset_name",
            "demo",
            "--skip_ai",
            "--no_auto_retry",
            "--no_interactive",
        ]
    ) == 0
    assert FakeMiceImputer.created == []
    assert captured_results[-1]["warnings"] == [
        "Lambda de severidad supero el umbral 0.30 para: global."
    ]


def test_no_interactive_without_goal_uses_prediction_without_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals_seen = _patch_lightweight_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *args, **kwargs: pytest.fail("input no debe llamarse en modo no interactivo"),
    )

    assert main(
        [
            "--input",
            "input.csv",
            "--dataset_name",
            "demo",
            "--skip_ai",
            "--no_interactive",
        ]
    ) == 0

    assert goals_seen == ["prediction"]


def test_explicit_goal_does_not_prompt_for_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals_seen = _patch_lightweight_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *args, **kwargs: pytest.fail("input no debe llamarse si goal y contexto vienen dados"),
    )

    assert main(
        [
            "--input",
            "input.csv",
            "--goal",
            "inference",
            "--domain_context",
            "datos de prueba",
            "--dataset_name",
            "demo",
            "--skip_ai",
        ]
    ) == 0

    assert goals_seen == ["inference"]


def test_no_interactive_does_not_prompt_for_suggestions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightweight_pipeline(monkeypatch, tmp_path)

    class FakeExplainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.client = types.SimpleNamespace(chat=lambda *args, **kwargs: "{}")

        def explain(self) -> dict[str, object]:
            return {
                "advertencias_y_sugerencias": {
                    "resumen": "hay sugerencias",
                    "acciones_sugeridas": ["Aumentar el numero de imputaciones a 10."],
                }
            }

    monkeypatch.setitem(
        sys.modules,
        "llm.explainer",
        types.SimpleNamespace(PipelineExplainer=FakeExplainer),
    )
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *args, **kwargs: pytest.fail("input no debe llamarse con --no_interactive"),
    )

    assert main(
        [
            "--input",
            "input.csv",
            "--goal",
            "inference",
            "--dataset_name",
            "demo",
            "--no_interactive",
        ]
    ) == 0


def test_invalid_suggestion_choice_does_not_break_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals_seen = _patch_lightweight_pipeline(monkeypatch, tmp_path)

    class FakeClient:
        def chat(self, *args: object, **kwargs: object) -> str:
            pytest.fail("no debe traducir una sugerencia invalida")

    class FakeExplainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.client = FakeClient()

        def explain(self) -> dict[str, object]:
            return {
                "advertencias_y_sugerencias": {
                    "resumen": "hay sugerencias",
                    "acciones_sugeridas": ["Aumentar el numero de imputaciones a 10."],
                }
            }

    inputs = iter(["99", "n"])
    monkeypatch.setitem(
        sys.modules,
        "llm.explainer",
        types.SimpleNamespace(PipelineExplainer=FakeExplainer),
    )
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: next(inputs))

    assert main(
        [
            "--input",
            "input.csv",
            "--goal",
            "inference",
            "--domain_context",
            "datos de prueba",
            "--dataset_name",
            "demo",
        ]
    ) == 0
    assert goals_seen == ["inference"]


def test_valid_suggestion_translation_triggers_recalculation_with_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightweight_pipeline(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    class FakeClient:
        def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            seen["translation_prompt"] = messages[0]["content"]
            return (
                '{"accion": "recalcular", "parametros": {"m": 10, '
                '"beta_vars": ["x"], "excluir_variables": ["y"]}}'
            )

    class FakeExplainer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.client = FakeClient()

        def explain(self) -> dict[str, object]:
            return {
                "advertencias_y_sugerencias": {
                    "resumen": "hay sugerencias",
                    "acciones_sugeridas": ["Excluir y y aumentar m a 10."],
                }
            }

    class FakeMiceImputer:
        def __init__(
            self,
            vars: list[str] | None = None,
            beta_vars: list[str] | None = None,
            m: int = 5,
        ) -> None:
            seen["m"] = m
            seen["vars"] = vars
            seen["beta_vars"] = beta_vars
            self.last_report = {
                "n_imputations": m,
                "severity": {"lambda": 0.1},
                "point_estimate": {"x": 1.0},
            }

        def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
            seen["columns_after_exclusion"] = list(df.columns)
            return df.fillna(0)

    import llm.explainer as real_explainer

    inputs = iter(["1", "n"])
    monkeypatch.setitem(
        sys.modules,
        "llm.explainer",
        types.SimpleNamespace(
            PipelineExplainer=FakeExplainer,
            _try_parse_action=real_explainer._try_parse_action,
        ),
    )
    monkeypatch.setattr(run_pipeline_module, "MiceImputer", FakeMiceImputer)
    monkeypatch.setattr(builtins, "input", lambda *args, **kwargs: next(inputs))

    assert main(
        [
            "--input",
            "input.csv",
            "--goal",
            "inference",
            "--domain_context",
            "datos de prueba",
            "--dataset_name",
            "demo",
        ]
    ) == 0

    assert seen["m"] == 10
    assert seen["beta_vars"] == ["x"]
    assert seen["columns_after_exclusion"] == ["x"]
    assert "excluir_variables" in seen["translation_prompt"]


def _patch_lightweight_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[str]:
    df = pd.DataFrame({"x": [1.0, None], "y": [2.0, 3.0]})
    goals_seen: list[str] = []

    class FakeDiagnosticoRunner:
        def __init__(self, plots_dir: Path) -> None:
            self.plots_dir = plots_dir

        def run(self, df: pd.DataFrame) -> dict[str, object]:
            return {"n_rows": len(df), "n_columns": len(df.columns)}

    class FakeImputationPipeline:
        def __init__(
            self,
            mice_vars: list[str] | None = None,
            beta_vars: list[str] | None = None,
        ) -> None:
            self.mice_vars = mice_vars
            self.beta_vars = beta_vars

        def run(self, df: pd.DataFrame, goal: str) -> dict[str, object]:
            goals_seen.append(goal)
            return {
                "decision": "regresion_estocastica",
                "reasoning": "mock",
                "criteria": {"max_pct": 50.0, "ratio": 1.0, "goal": goal},
                "imputed_data": df.fillna(0),
                "imputer_report": None,
                "warnings": [],
            }

        def _build_lambda_warnings(
            self,
            report: dict[str, object] | None,
        ) -> list[str]:
            return []

    output_counter = {"value": 0}

    def fake_make_output_dir(dataset_name: str) -> Path:
        output_counter["value"] += 1
        output_dir = tmp_path / "outputs" / f"{dataset_name}_{output_counter['value']}"
        output_dir.mkdir(parents=True)
        return output_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pipeline_module, "_make_output_dir", fake_make_output_dir)
    monkeypatch.setattr(run_pipeline_module, "load_file", lambda path: df.copy())
    monkeypatch.setattr(
        run_pipeline_module,
        "build_profile",
        lambda df: {"n_rows": len(df), "n_columns": len(df.columns)},
    )
    monkeypatch.setattr(run_pipeline_module, "standardize", lambda df, profile: df)
    monkeypatch.setattr(run_pipeline_module, "detect_id_columns", lambda df: [])
    monkeypatch.setattr(
        run_pipeline_module,
        "drop_rows_without_information",
        lambda df, id_columns: (df, []),
    )
    monkeypatch.setattr(run_pipeline_module, "DiagnosticoRunner", FakeDiagnosticoRunner)
    monkeypatch.setattr(run_pipeline_module, "ImputationPipeline", FakeImputationPipeline)
    monkeypatch.setattr(
        run_pipeline_module,
        "build_comparison_plots",
        lambda before, after, missing_vars: {},
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "build_report_json",
        lambda diagnostico, pipeline_result, **kwargs: {
            "metadata": {"dataset_name": "demo", "n_rows": 2, "n_columns": 2},
            "exploracion_previa": diagnostico,
            "decision": {"metodo": pipeline_result["decision"]},
            "imputacion": pipeline_result["imputer_report"],
            "advertencias": pipeline_result["warnings"],
        },
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "generate_html",
        lambda report_json_path, output_html_path: None,
    )
    return goals_seen
