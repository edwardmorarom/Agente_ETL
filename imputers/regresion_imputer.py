from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import pandas as pd

from imputers.base import BaseImputer, ImputerExecutionError, format_command


class RegresionImputer(BaseImputer):
    def __init__(
        self,
        method: str = "stochastic_regression",
        models: Path | str | None = None,
        seed: int = 123,
        rscript_path: str = "Rscript",
        script_path: Path | None = None,
    ) -> None:
        self.method = method
        self.models = Path(models) if models is not None else None
        self.seed = seed
        self.rscript_path = rscript_path
        self.script_path = script_path or (
            Path(__file__).resolve().parents[1]
            / "r_scripts"
            / "regresion_imputer.R"
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            output_csv = tmp_path / "output.csv"

            df.to_csv(input_csv, index=False)
            command = self._build_command(input_csv, output_csv)
            self._run(command)

            return pd.read_csv(output_csv)

    def _build_command(self, input_csv: Path, output_csv: Path) -> list[str]:
        command = [
            self.rscript_path,
            str(self.script_path),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--method",
            self.method,
            "--seed",
            str(self.seed),
        ]
        if self.models is not None:
            command.extend(["--models", str(self.models)])
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
                "El script R de regresión falló. "
                f"Comando intentado: {format_command(command)}. "
                f"Detalle: {details}"
            ) from exc
