from __future__ import annotations

import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from imputers.base import BaseImputer, ImputerExecutionError, format_command


class MiceImputer(BaseImputer):
    def __init__(
        self,
        vars: list[str] | None = None,
        m: int = 5,
        maxit: int = 5,
        seed: int = 123,
        rscript_path: str = "Rscript",
        script_path: Path | None = None,
    ) -> None:
        self.vars = vars
        self.m = m
        self.maxit = maxit
        self.seed = seed
        self.rscript_path = rscript_path
        self.script_path = script_path or (
            Path(__file__).resolve().parents[1] / "r_scripts" / "mice_imputer.R"
        )
        self.last_report: dict[str, Any] | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            output_csv = tmp_path / "output.csv"
            output_json = tmp_path / "report.json"

            df.to_csv(input_csv, index=False)
            command = self._build_command(input_csv, output_csv, output_json)
            self._run(command)

            result = pd.read_csv(output_csv)
            with open(output_json, "r", encoding="utf-8") as report_file:
                self.last_report = json.load(report_file)

            return result

    def _build_command(
        self,
        input_csv: Path,
        output_csv: Path,
        output_json: Path,
    ) -> list[str]:
        command = [
            self.rscript_path,
            str(self.script_path),
            "--input",
            str(input_csv),
            "--output_data",
            str(output_csv),
            "--output_json",
            str(output_json),
            "--m",
            str(self.m),
            "--maxit",
            str(self.maxit),
            "--seed",
            str(self.seed),
        ]
        if self.vars:
            command.extend(["--vars", ",".join(self.vars)])
        return command

    def _run(self, command: list[str]) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise ImputerExecutionError(
                f"No se pudo encontrar Rscript. Comando intentado: {format_command(command)}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or str(exc)).strip()
            raise ImputerExecutionError(
                "El script R de MICE falló. "
                f"Comando intentado: {format_command(command)}. "
                f"Detalle: {details}"
            ) from exc
