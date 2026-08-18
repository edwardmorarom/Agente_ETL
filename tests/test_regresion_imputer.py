from __future__ import annotations

from pathlib import Path
import subprocess

import pandas as pd
import pytest

from imputers.base import ImputerExecutionError
from imputers.regresion_imputer import RegresionImputer


def test_regresion_imputer_runs_r_script(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0], "z": [10.0, 11.0, 12.0]})
    seen_command: list[str] = []

    def fake_run(
        command: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        assert encoding == "utf-8"
        assert errors == "replace"
        seen_command.extend(command)
        input_csv = Path(command[command.index("--input") + 1])
        output_csv = Path(command[command.index("--output") + 1])

        written = pd.read_csv(input_csv)
        assert list(written.columns) == ["x", "z"]
        written.fillna({"x": 2.0}).to_csv(output_csv, index=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    imputer = RegresionImputer(method="stochastic_mean", seed=7)
    result = imputer.fit_transform(df)

    assert seen_command[:2] == ["Rscript", str(imputer.script_path)]
    assert seen_command[seen_command.index("--method") + 1] == "stochastic_mean"
    assert seen_command[seen_command.index("--seed") + 1] == "7"
    assert result.isna().sum().sum() == 0


def test_regresion_imputer_passes_models_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models = tmp_path / "models.txt"
    models.write_text("x ~ z", encoding="utf-8")
    seen_command: list[str] = []

    def fake_run(
        command: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        assert encoding == "utf-8"
        assert errors == "replace"
        seen_command.extend(command)
        output_csv = Path(command[command.index("--output") + 1])
        pd.DataFrame({"x": [1.0], "z": [2.0]}).to_csv(output_csv, index=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    RegresionImputer(models=models).fit_transform(pd.DataFrame({"x": [None], "z": [2.0]}))

    assert seen_command[seen_command.index("--models") + 1] == str(models)


def test_regresion_imputer_reports_missing_rscript(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        command: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ImputerExecutionError, match="Comando intentado: Rscript"):
        RegresionImputer().fit_transform(pd.DataFrame({"x": [1.0, None]}))


def test_regresion_imputer_reports_r_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        command: list[str],
        check: bool,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command, stderr="fallo R")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ImputerExecutionError, match="El script R de regresión falló"):
        RegresionImputer().fit_transform(pd.DataFrame({"x": [1.0, None]}))
