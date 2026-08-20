"""
Este modulo nunca debe recibir ni enviar el DataFrame imputado ni
filas de datos reales al LLM. Solo se comparten metadatos agregados
del pipeline: decision, razonamiento, advertencias y el reporte
estadistico de severidad. Esto es una regla de privacidad, no una
opcion de configuracion.
"""

from __future__ import annotations

from pprint import pformat
from typing import Any

from llm.client import LLMClient, get_llm_client


class PipelineExplainer:
    def __init__(
        self,
        pipeline_result: dict[str, Any],
        client: LLMClient | None = None,
    ) -> None:
        self.pipeline_result = pipeline_result
        self.client = client or get_llm_client()
        self.history: list[dict[str, str]] = []

    def explain(self) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente que explica en espanol claro y sencillo "
                    "los resultados de un pipeline de imputacion de datos "
                    "faltantes. No te limites a reportar los numeros: razona "
                    "sobre que implica cada resultado. Si un supuesto estadistico "
                    "no se cumple, explica el riesgo practico que eso representa "
                    "para las conclusiones del usuario, no solo que 'no se cumplio'."
                ),
            },
            {
                "role": "user",
                "content": self._build_summary_message(),
            },
        ]

        response = self.client.chat(
            messages,
            generation_options={"temperature": 0.2, "num_predict": 400},
        )
        self.history.extend(
            [
                messages[0],
                messages[1],
                {"role": "assistant", "content": response},
            ]
        )
        return response

    def ask(self, question: str) -> str:
        if not self.history:
            self.explain()

        if len(self.history) > 12:
            self.history = self.history[:2] + self.history[-8:]

        self.history.append({"role": "user", "content": question})
        response = self.client.chat(
            self.history,
            generation_options={"temperature": 0.4, "num_predict": 350},
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    def _build_summary_message(self) -> str:
        lines = [
            "Explica el resultado del pipeline usando solo estos metadatos:",
            f"Decision: {self.pipeline_result.get('decision')}",
            f"Razonamiento: {self.pipeline_result.get('reasoning')}",
            f"Advertencias: {self._format_value(self.pipeline_result.get('warnings', []))}",
        ]

        imputer_report = self.pipeline_result.get("imputer_report")
        if imputer_report is not None:
            lines.append(f"Reporte del imputador: {self._format_value(imputer_report)}")

        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        return pformat(value, width=88, sort_dicts=True)
