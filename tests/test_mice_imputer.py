from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from imputers.base import ImputerExecutionError
from imputers.mice_imputer import MiceImputer


def test_mice_imputer_runs_r_script_and_keeps_report(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [2.0, 4.0, None]})
    report = {"case": "multivariate", "n_imputations": 2}
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
        output_csv = Path(command[command.index("--output_data") + 1])
        output_json = Path(command[command.index("--output_json") + 1])

        written = pd.read_csv(input_csv)
        assert list(written.columns) == ["x", "y"]
        written.fillna({"x": 2.0, "y": 3.0}).to_csv(output_csv, index=False)
        output_json.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    imputer = MiceImputer(vars=["x", "y"], m=2, maxit=3, seed=99)
    result = imputer.fit_transform(df)

    assert seen_command[:2] == ["Rscript", str(imputer.script_path)]
    assert "--output_data" in seen_command
    assert "--output_json" in seen_command
    assert seen_command[seen_command.index("--vars") + 1] == "x,y"
    assert imputer.last_report == report
    assert result.isna().sum().sum() == 0


def test_mice_imputer_reports_missing_rscript(monkeypatch: pytest.MonkeyPatch) -> None:
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
        MiceImputer().fit_transform(pd.DataFrame({"x": [1.0, None]}))


def test_mice_imputer_reports_r_failure(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with pytest.raises(ImputerExecutionError, match="El script R de MICE falló"):
        MiceImputer().fit_transform(pd.DataFrame({"x": [1.0, None]}))
