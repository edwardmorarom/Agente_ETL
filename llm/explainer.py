"""
Este modulo nunca debe recibir ni enviar el DataFrame imputado ni
filas de datos reales al LLM. Solo se comparte el reporte consolidado
del pipeline, compuesto por metadatos, resumenes agregados, diagnostico,
decision, resultados estadisticos y advertencias. Esto es una regla de
privacidad, no una opcion de configuracion.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from llm.client import LLMClient, get_llm_client


SYSTEM_PROMPT = (
    "Eres un asistente que interpreta en espanol claro y sencillo el "
    "resultado completo de un pipeline de imputacion de datos faltantes, "
    "a partir de un JSON con toda la informacion disponible. REGLA "
    "ABSOLUTA: Usa UNICAMENTE los valores, nombres de pruebas y metricas "
    "que aparecen LITERALMENTE en el JSON que se te proporciona. NUNCA "
    "inventes nombres de pruebas estadisticas, porcentajes, valores p, ni "
    "ninguna cifra que no este explicitamente en el JSON. Por ejemplo, si "
    "el JSON menciona una prueba de Anderson-Darling, NO la llames "
    "Kolmogorov-Smirnov ni Shapiro-Wilk. Si el JSON no incluye un dato que "
    "el usuario pregunta, responde honestamente 'ese dato no aparece en el "
    "reporte' en vez de inventar un valor plausible. Alucinar datos falsos "
    "es el peor error que puedes cometer en esta tarea, mas grave que decir "
    "'no lo se'. El JSON contiene diagnostico "
    "previo, metodo elegido, resultados de la imputacion, cumplimiento de "
    "supuestos estadisticos, y cualquier analisis adicional presente en el "
    "JSON, como ajuste de distribuciones parametricas si aplica). No te "
    "limites a un solo indicador: revisa TODOS los resultados de supuestos "
    "y pruebas presentes, y si varios de ellos senalan problemas (baja "
    "convergencia, mala comparacion de distribuciones, severidad alta, mal "
    "ajuste de distribuciones, etc.), senala el patron general y explica el "
    "riesgo practico para las conclusiones del usuario. Si todo se ve bien, "
    "dilo tambien con confianza. Razona sobre las implicaciones, no solo "
    "reportes los numeros. Si se te da contexto sobre el dominio de los "
    "datos, incorporalo en tu analisis para que sea mas relevante. Si NO se "
    "te da contexto de dominio, analiza unicamente con base en los patrones "
    "estadisticos, sin asumir de que trata el dataset ni inventar un dominio. "
    "IMPORTANTE: no tienes acceso a ningun archivo, documento, HTML, ruta "
    "de disco, ni a nada fuera de esta conversacion. Toda tu informacion "
    "proviene UNICAMENTE del resumen de datos que se te comparte en este "
    "mensaje. Si el usuario te pregunta si puedes leer un archivo, documento "
    "o el reporte HTML directamente, responde con honestidad que no tienes "
    "acceso a archivos, pero que si conoces el contenido analitico que se te "
    "compartio y puedes responder con base en eso."
)


class PipelineExplainer:
    def __init__(
        self,
        report: dict[str, Any],
        client: LLMClient | None = None,
        domain_context: str | None = None,
    ) -> None:
        self.report = _sanitize_for_llm(report)
        self.client = client or get_llm_client()
        self.domain_context = domain_context
        self.history: list[dict[str, str]] = []
        self.last_structured_explanation: dict[str, str] = {}

    def explain(self) -> dict[str, str]:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._build_summary_message(),
            },
        ]

        raw_response = self.client.chat(
            messages,
            generation_options={"temperature": 0.1, "num_predict": 1200, "num_ctx": 8192},
        )
        response = _parse_structured_response(raw_response)
        self.last_structured_explanation = response
        assistant_text = _structured_response_to_text(response)
        self.history.extend(
            [
                messages[0],
                messages[1],
                {"role": "assistant", "content": assistant_text},
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
            generation_options={"temperature": 0.15, "num_predict": 700, "num_ctx": 8192},
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    def _build_summary_message(self) -> str:
        report_json = json.dumps(
            self.report,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        lines = [
            "Explica el resultado completo del pipeline a partir de este JSON:",
            report_json,
            (
                "Responde UNICAMENTE con un JSON valido (sin texto antes ni "
                "despues, sin bloques de codigo markdown), con esta estructura "
                'exacta:\n{\n  "general": "resumen ejecutivo de 3-4 oraciones '
                'sobre el resultado completo",\n  "exploracion": "explicacion '
                'de 2-3 oraciones sobre el diagnostico previo de faltantes '
                '(patron, test de Little, EM)",\n  "resumen_descriptivo": '
                '"explicacion de 1-2 oraciones sobre como cambiaron las '
                'variables antes/despues de imputar",\n  "decision": '
                '"explicacion de 1-2 oraciones sobre por que se eligio ese '
                'metodo",\n  "imputacion": "explicacion de 2-3 oraciones '
                'sobre los resultados de la imputacion (point_estimate, '
                'severidad, y ajuste de distribucion Beta si esta presente)",\n'
                '  "supuestos": "explicacion de 2-3 oraciones sobre que '
                'supuestos se cumplieron o no, y que implica eso",\n  '
                '"advertencias": "explicacion breve de las advertencias, o '
                'una frase confirmando que no hubo ninguna"\n}\nCada valor '
                "debe basarse SOLO en los datos reales del JSON de arriba, "
                "nunca inventes contenido para rellenar una seccion si no hay "
                "informacion suficiente - en ese caso escribe una frase breve "
                "indicandolo, por ejemplo 'No hay informacion adicional "
                "relevante en esta seccion.'"
            ),
        ]
        if self.domain_context is not None and self.domain_context.strip():
            lines.insert(
                0,
                "Contexto del dataset proporcionado por el usuario: "
                f"{self.domain_context}",
            )

        return "\n".join(lines)


def _sanitize_for_llm(report: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(report)
    sanitized.pop("imputed_data", None)
    exploration = sanitized.get("exploracion_previa")
    if not isinstance(exploration, dict):
        return sanitized

    exploration.pop("comparison_plots", None)

    plots_generated = exploration.get("plots_generated")
    if isinstance(plots_generated, list):
        exploration["plots_generated"] = [
            Path(str(path)).name for path in plots_generated
        ]

    imputation = sanitized.get("imputacion")
    if not isinstance(imputation, dict):
        return sanitized

    convergence = (
        imputation.get("assumptions", {})
        .get("convergence")
        if isinstance(imputation.get("assumptions"), dict)
        else None
    )
    if isinstance(convergence, dict):
        for payload in convergence.values():
            if isinstance(payload, dict):
                payload.pop("chain_mean", None)

    covariance_estimate = imputation.get("covariance_estimate")
    if isinstance(covariance_estimate, dict) and len(covariance_estimate) > 6:
        # Si en el futuro estas matrices crecen demasiado para el contexto del LLM,
        # aqui se podria resumir sin cambiar la estructura del reporte original.
        pass

    return sanitized


def _parse_structured_response(raw_text: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"general": raw_text}
        try:
            parsed = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return {"general": raw_text}

    if not isinstance(parsed, dict):
        return {"general": raw_text}

    return {str(key): str(value) for key, value in parsed.items()}


def _structured_response_to_text(response: dict[str, str]) -> str:
    labels = {
        "general": "General",
        "exploracion": "Exploracion",
        "resumen_descriptivo": "Resumen descriptivo",
        "decision": "Decision",
        "imputacion": "Imputacion",
        "supuestos": "Supuestos",
        "advertencias": "Advertencias",
    }
    pieces = []
    for key, label in labels.items():
        value = response.get(key)
        if value:
            pieces.append(f"{label}:\n{value}")
    return "\n\n".join(pieces) if pieces else response.get("general", "")
