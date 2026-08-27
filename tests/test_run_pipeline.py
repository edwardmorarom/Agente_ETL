from __future__ import annotations

import builtins
import json
from pathlib import Path
import shutil
import sys
import types

import pandas as pd
import pytest

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
