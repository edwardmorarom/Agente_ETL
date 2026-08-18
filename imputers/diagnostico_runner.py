from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
import warnings

import pandas as pd

from imputers.base import ImputerExecutionError, format_command


class DiagnosticoRunner:
    def __init__(
        self,
        plots_dir: Path | str | None = None,
        rscript_path: str = "Rscript",
        script_path: Path | None = None,
    ) -> None:
        self.plots_dir = Path(plots_dir) if plots_dir is not None else Path("diagnostico_output")
        self.rscript_path = rscript_path
        self.script_path = script_path or (
            Path(__file__).resolve().parents[1] / "r_scripts" / "diagnostico.R"
        )

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "input.csv"
            output_json = tmp_path / "diagnostico.json"
            output_plots_dir = tmp_path / "plots"

            df.to_csv(input_csv, index=False)
            command = self._build_command(input_csv, output_json, output_plots_dir)
            self._run(command)

            with open(output_json, "r", encoding="utf-8") as report_file:
                report: dict[str, Any] = json.load(report_file)

            plots_generated = self._copy_plots(output_plots_dir)
            if not plots_generated:
                warnings.warn(
                    "diagnostico.R no generó ningún gráfico PNG, revisa su salida"
                )
            report["plots_generated"] = plots_generated
            return report

    def _build_command(
        self,
        input_csv: Path,
        output_json: Path,
        output_plots_dir: Path,
    ) -> list[str]:
        return [
            self.rscript_path,
            str(self.script_path),
            "--input",
            str(input_csv),
            "--output_json",
            str(output_json),
            "--output_plots_dir",
            str(output_plots_dir),
        ]

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
                "El script R de diagnostico fallo. "
                f"Comando intentado: {format_command(command)}. "
                f"Detalle: {details}"
            ) from exc

    def _copy_plots(self, source_dir: Path) -> list[str]:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        copied_paths: list[str] = []

        for source_path in sorted(source_dir.glob("*.png")):
            destination = self.plots_dir / source_path.name
            shutil.copy2(source_path, destination)
            copied_paths.append(str(destination))

        return copied_paths
