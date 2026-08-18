from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from imputers.base import ImputerExecutionError
from imputers.diagnostico_runner import DiagnosticoRunner


def test_diagnostico_runner_runs_r_script_and_copies_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [2.0, 4.0, None]})
    plots_dir = tmp_path / "persistent_plots"
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
        output_json = Path(command[command.index("--output_json") + 1])
        output_plots_dir = Path(command[command.index("--output_plots_dir") + 1])

        written = pd.read_csv(input_csv)
        assert list(written.columns) == ["x", "y"]

        output_plots_dir.mkdir(parents=True, exist_ok=True)
        (output_plots_dir / "gg_miss_var.png").write_bytes(b"png data")
        output_json.write_text(
            json.dumps({"mcar_test_naniar": {"p_value": 0.5}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = DiagnosticoRunner(plots_dir=plots_dir)
    result = runner.run(df)

    assert seen_command[:2] == ["Rscript", str(runner.script_path)]
    assert "--output_json" in seen_command
    assert "--output_plots_dir" in seen_command
    assert result["mcar_test_naniar"]["p_value"] == 0.5
    assert result["plots_generated"] == [str(plots_dir / "gg_miss_var.png")]
    assert (plots_dir / "gg_miss_var.png").read_bytes() == b"png data"


def test_diagnostico_runner_reports_missing_rscript(monkeypatch: pytest.MonkeyPatch) -> None:
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
        DiagnosticoRunner().run(pd.DataFrame({"x": [1.0, None]}))


def test_diagnostico_runner_reports_r_failure(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with pytest.raises(ImputerExecutionError, match="El script R de diagnostico fallo"):
        DiagnosticoRunner().run(pd.DataFrame({"x": [1.0, None]}))


@pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript no esta disponible en el sistema.",
)
def test_diagnostico_runner_runs_real_r_script(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, None, 3.5, 4.2, 5.8, 7.1],
            "y": [2.2, 3.9, None, 8.4, 9.7, 13.1],
            "z": [5.0, 1.4, 6.2, None, 3.3, 8.9],
        }
    )
    plots_dir = tmp_path / "diagnostico_plots"

    result = DiagnosticoRunner(plots_dir=plots_dir).run(df)

    assert "mcar_test_naniar" in result
    assert "little_test_manual" in result
    assert result["plots_generated"]
    assert all(Path(path).exists() for path in result["plots_generated"])
    assert all(Path(path).parent == plots_dir for path in result["plots_generated"])
