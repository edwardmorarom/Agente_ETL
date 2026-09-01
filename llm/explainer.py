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
from typing import Any, Callable

from llm.client import LLMClient, get_llm_client


BASE_SECTION_PROMPT = (
    "Eres un asistente que interpreta en espanol claro y sencillo UNA PARTE "
    "especifica del resultado de un pipeline de imputacion de datos faltantes. "
    "REGLA ABSOLUTA: usa UNICAMENTE los valores que aparecen LITERALMENTE en "
    "el JSON que se te da en este mensaje. NUNCA inventes numeros, nombres de "
    "pruebas, ni datos que no esten aqui. Si no hay suficiente informacion "
    "para algo, dilo honestamente en vez de inventar. Responde en 2-4 "
    "oraciones de texto plano, sin JSON, sin markdown, sin encabezados. "
    "Cuando menciones una metrica estadistica (como lambda, r, gamma, un "
    "p-valor, o un parametro de una distribucion), explica en terminos "
    "practicos que significa ese valor especifico y que implica para la "
    "confiabilidad de los resultados. IMPORTANTE: no tienes acceso a ningun "
    "archivo, documento, HTML, ruta de disco, ni a nada fuera de esta "
    "conversacion. Toda tu informacion proviene UNICAMENTE del JSON que se te "
    "comparte en este mensaje."
)

ASK_TOOL_PROMPT = (
    "Ademas de responder preguntas sobre el reporte ya generado, tienes la "
    "capacidad de EJECUTAR nuevos analisis si el usuario lo pide "
    "explicitamente (por ejemplo, pedir mas imputaciones, cambiar el objetivo "
    "de inferencia a prediccion o viceversa, o probar el ajuste Beta en otra "
    "variable numerica). Cuando el usuario pida ALGO QUE REQUIERA RECALCULAR "
    "(no solo explicar lo que ya existe), responde UNICAMENTE con un JSON en "
    "este formato exacto, sin texto adicional: "
    '{"accion": "recalcular", "parametros": {"m": numero_opcional, '
    '"goal": "inference_o_prediction_opcional", '
    '"beta_vars": ["lista_opcional_de_variables"], '
    '"excluir_variables": ["lista_opcional_de_variables"]}}. Incluye SOLO los '
    "parametros que el usuario pidio cambiar, omite los demas. Si la pregunta "
    "es solo informativa (no requiere recalcular), responde normalmente en "
    "texto explicativo, NUNCA en este formato JSON. IMPORTANTE sobre "
    "comparaciones antes/despues de un recalculo: cada corrida de MICE usa "
    "una semilla aleatoria distinta, por lo que una diferencia entre el "
    "resultado original y el recalculado PUEDE deberse a variabilidad "
    "aleatoria normal entre corridas, no necesariamente a un efecto real del "
    "parametro que se cambio. NUNCA afirmes de forma categorica que un "
    "parametro 'empeora' o 'mejora' los resultados basandote en una sola "
    "comparacion de 2 corridas. En su lugar, describe la diferencia observada "
    "con cautela (ej: 'en esta corrida especifica, lambda subio a X, aunque "
    "esto podria deberse tanto al cambio de parametro como a la variabilidad "
    "normal entre imputaciones distintas') y aclara que confirmar una "
    "tendencia real requeriria repetir la comparacion varias veces."
)

EMPTY_CHAT_FALLBACK = (
    "No pude generar una explicacion del recalculo, pero la operacion se "
    "ejecuto correctamente. Revisa el reporte actualizado o intenta "
    "reformular tu pregunta."
)


class PipelineExplainer:
    def __init__(
        self,
        report: dict[str, Any],
        client: LLMClient | None = None,
        domain_context: str | None = None,
        tool_executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.report = _sanitize_for_llm(report)
        self.client = client or get_llm_client()
        self.domain_context = domain_context
        self.tool_executor = tool_executor
        self.history: list[dict[str, str]] = []
        self.last_structured_explanation: dict[str, str] = {}

    def explain(self) -> dict[str, str]:
        counts = _compute_assumption_counts(self.report.get("imputacion"))
        resultado: dict[str, str] = {}
        secciones = [
            ("general", self._payload_general(counts)),
            ("exploracion", self._payload_exploracion()),
            ("resumen_descriptivo", self._payload_resumen_descriptivo()),
            ("decision", self._payload_decision()),
            ("imputacion", self._payload_imputacion()),
            ("supuestos", self._payload_supuestos(counts)),
            ("advertencias_y_sugerencias", self._payload_advertencias(counts)),
        ]

        for nombre, payload in secciones:
            messages = [
                {"role": "system", "content": self._system_prompt_seccion(nombre)},
                {"role": "user", "content": self._build_section_message(nombre, payload)},
            ]
            respuesta = self.client.chat(
                messages,
                generation_options={
                    "temperature": 0.1,
                    "num_predict": 400,
                    "num_ctx": 4096,
                },
            )
            texto = respuesta.strip()
            if not texto:
                texto = self._fallback_section_text(nombre, payload)
            resultado[nombre] = texto

        self.last_structured_explanation = resultado
        texto_combinado = "\n\n".join(
            f"## {nombre.replace('_', ' ').title()}\n{texto}"
            for nombre, texto in resultado.items()
        )
        self.history = [
            {"role": "system", "content": self._system_prompt_chat()},
            {"role": "assistant", "content": texto_combinado},
        ]
        return resultado

    def ask(self, question: str) -> str:
        if not self.history:
            self.explain()

        if len(self.history) > 12:
            self.history = self.history[:2] + self.history[-8:]

        self.history.append({"role": "user", "content": question})
        response = self.client.chat(
            self._messages_for_ask(),
            generation_options={"temperature": 0.15, "num_predict": 700, "num_ctx": 8192},
        )
        action = _try_parse_action(response)
        if action is not None:
            self.history.append({"role": "assistant", "content": response})
            if self.tool_executor is None:
                unavailable = (
                    "La capacidad de recalcular no esta disponible en este modo."
                )
                self.history.append({"role": "assistant", "content": unavailable})
                return unavailable

            parametros = action["parametros"]
            new_result = _sanitize_for_llm(self.tool_executor(parametros))
            summary_message = (
                "Resultado del recalculo solicitado por el usuario. "
                "Explicalo en texto natural, comparando con el resultado "
                "anterior cuando sea posible, sin inventar datos:\n"
                f"{json.dumps(new_result, indent=2, ensure_ascii=False, default=str)}"
            )
            self.history.append({"role": "user", "content": summary_message})
            final_response = self.client.chat(
                self._messages_for_ask(),
                generation_options={
                    "temperature": 0.15,
                    "num_predict": 1000,
                    "num_ctx": 8192,
                },
            )
            final_text = final_response.strip() or EMPTY_CHAT_FALLBACK
            self.history.append({"role": "assistant", "content": final_text})
            return final_text

        response_text = response.strip() or EMPTY_CHAT_FALLBACK
        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def _messages_for_ask(self) -> list[dict[str, str]]:
        messages = [message.copy() for message in self.history]
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{ASK_TOOL_PROMPT}"
            return messages
        return [{"role": "system", "content": ASK_TOOL_PROMPT}, *messages]

    def _build_section_message(self, nombre: str, payload: dict[str, Any]) -> str:
        return (
            f"Seccion: {nombre}\n"
            "Datos disponibles para esta seccion:\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False, default=str)}"
        )

    def _payload_general(self, counts: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": self.report.get("decision"),
            "metadata": self.report.get("metadata"),
            "advertencias_automaticas_count": len(self.report.get("advertencias", [])),
            "resumen_supuestos": counts,
        }

    def _payload_exploracion(self) -> dict[str, Any]:
        return self.report.get("exploracion_previa", {})

    def _payload_resumen_descriptivo(self) -> dict[str, Any]:
        return self.report.get("resumen_descriptivo", {})

    def _payload_decision(self) -> dict[str, Any]:
        return self.report.get("decision", {})

    def _payload_imputacion(self) -> dict[str, Any]:
        imputacion = self.report.get("imputacion")
        if not isinstance(imputacion, dict):
            return {}

        payload = {
            key: imputacion[key]
            for key in ("point_estimate", "severity")
            if key in imputacion
        }
        beta_fit = imputacion.get("beta_distribution_fit")
        if isinstance(beta_fit, dict):
            payload["beta_distribution_fit"] = {
                variable: {
                    key: value
                    for key, value in fit.items()
                    if key in ("shape1_pooled", "shape2_pooled", "severity_joint")
                }
                for variable, fit in beta_fit.items()
                if isinstance(fit, dict)
            }
        return payload

    def _payload_supuestos(self, counts: dict[str, Any]) -> dict[str, Any]:
        imputacion = self.report.get("imputacion")
        assumptions = (
            imputacion.get("assumptions", {})
            if isinstance(imputacion, dict)
            else {}
        )
        convergence = assumptions.get("convergence", {})
        distribution = assumptions.get("distribution_comparison", {})
        return {
            "conteos": counts,
            "detalle_convergence": {
                variable: payload.get("converged")
                for variable, payload in convergence.items()
                if isinstance(payload, dict)
            } if isinstance(convergence, dict) else {},
            "detalle_distribution": {
                variable: payload.get("meets_assumption")
                for variable, payload in distribution.items()
                if isinstance(payload, dict)
            } if isinstance(distribution, dict) else {},
        }

    def _payload_advertencias(self, counts: dict[str, Any]) -> dict[str, Any]:
        return {
            "conteos": counts,
            "advertencias_del_pipeline": self.report.get("advertencias", []),
        }

    def _system_prompt_seccion(self, nombre: str) -> str:
        instructions = {
            "general": (
                "Resume el estado general del pipeline: dataset, metodo elegido "
                "y presencia o ausencia de advertencias. No afirmes que la "
                "ejecucion fue 'limpia' o 'sin problemas' basandote UNICAMENTE "
                "en si el pipeline genero advertencias automaticas. Revisa "
                "tambien 'resumen_supuestos': si hay variables que no "
                "convergieron, no cumplieron la comparacion de distribucion, o "
                "tienen un lambda alto en el ajuste Beta, menciona brevemente "
                "que existen hallazgos relevantes en el detalle (sin repetir "
                "el conteo exacto, eso se explica en otra seccion), en vez de "
                "dar una conclusion general demasiado optimista o "
                "contradictoria con el resto del reporte."
            ),
            "exploracion": (
                "Explica el diagnostico previo de faltantes y que sugiere sobre "
                "el patron de ausencia de datos."
            ),
            "resumen_descriptivo": (
                "Explica como cambiaron las estadisticas agregadas antes y "
                "despues de imputar, sin inferir filas individuales."
            ),
            "decision": (
                "Explica por que la decision metodologica es coherente con los "
                "criterios disponibles."
            ),
            "imputacion": (
                "Explica los resultados de imputacion, la severidad y el ajuste "
                "Beta si aparece en el JSON."
            ),
            "supuestos": (
                "Explica que proporcion de variables cumplio cada supuesto "
                "usando los conteos dados, y que implica eso para la "
                "confiabilidad de las conclusiones."
            ),
            "advertencias_y_sugerencias": (
                "Con base en los conteos y advertencias dados, da 1-3 "
                "sugerencias concretas y accionables. Si todo se ve bien, dilo "
                "con confianza. Las acciones_sugeridas que propongas DEBEN "
                "ser unicamente de estos 4 tipos, ya que son las unicas que el "
                "sistema puede ejecutar automaticamente: 1. Aumentar el numero "
                "de imputaciones (especifica un numero razonable). 2. Cambiar "
                "el objetivo del analisis (inferencia <-> prediccion). 3. "
                "Agregar ajuste de distribucion Beta a una variable especifica "
                "que parezca un porcentaje o proporcion. 4. Excluir una "
                "variable especifica del analisis por tener demasiados "
                "problemas (mala convergencia y/o mal ajuste de distribucion). "
                "NO sugieras cambios de metodo de imputacion, librerias, ni "
                "nada que estos 4 tipos no cubran. Si no hay ninguna accion de "
                "estos 4 tipos que aplique, deja acciones_sugeridas como una "
                "lista vacia."
            ),
        }
        prompt = f"{BASE_SECTION_PROMPT} {instructions.get(nombre, '')}"
        if self.domain_context is not None and self.domain_context.strip():
            prompt = f"{prompt} Contexto del dataset: {self.domain_context}."
        return prompt

    def _system_prompt_chat(self) -> str:
        prompt = (
            f"{BASE_SECTION_PROMPT} Ahora estas en modo conversacional: responde "
            "preguntas sobre las explicaciones ya generadas y el reporte "
            "resumido en el historial. Si el usuario pide recalcular, sigue las "
            "instrucciones de herramienta que se agregan al mensaje de sistema."
        )
        if self.domain_context is not None and self.domain_context.strip():
            prompt = f"{prompt} Contexto del dataset: {self.domain_context}."
        return prompt

    def _fallback_section_text(self, nombre: str, payload: dict[str, Any]) -> str:
        if not payload:
            return (
                "No se genero una interpretacion de IA para esta seccion y no "
                "hay datos suficientes en el reporte para resumirla."
            )
        if nombre == "imputacion":
            parts = []
            if "point_estimate" in payload:
                parts.append("estimaciones puntuales")
            if "severity" in payload:
                parts.append("metricas de severidad")
            if "beta_distribution_fit" in payload:
                parts.append("ajuste de distribucion Beta")
            details = ", ".join(parts) if parts else "resultados de imputacion"
            return (
                "No se genero una interpretacion de IA para esta seccion, pero "
                f"el reporte contiene {details}. Revisa las tablas de esta "
                "seccion para evaluar magnitudes y severidad sin inventar "
                "conclusiones adicionales."
            )
        return (
            "No se genero una interpretacion de IA para esta seccion. Los datos "
            "estructurados de la seccion se mantienen disponibles en el reporte."
        )


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


def _compute_assumption_counts(imputacion: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(imputacion, dict):
        return {}

    assumptions = imputacion.get("assumptions", {})
    counts: dict[str, Any] = {}
    if isinstance(assumptions, dict):
        convergence = assumptions.get("convergence", {})
        if isinstance(convergence, dict):
            convergence_counts = {"convergio": 0, "no_convergio": 0, "no_aplica": 0}
            for payload in convergence.values():
                value = payload.get("converged") if isinstance(payload, dict) else None
                if value is True:
                    convergence_counts["convergio"] += 1
                elif value is False:
                    convergence_counts["no_convergio"] += 1
                else:
                    convergence_counts["no_aplica"] += 1
            counts["convergence"] = convergence_counts

        distribution = assumptions.get("distribution_comparison", {})
        if isinstance(distribution, dict):
            counts["distribution_comparison"] = _count_meets_assumption(
                distribution.values()
            )

        normality_payloads = []
        homoscedasticity_payloads = []
        for key, payload in assumptions.items():
            if key in ("convergence", "distribution_comparison"):
                continue
            if not isinstance(payload, dict):
                continue
            if "normality" in payload:
                normality_payloads.append(payload.get("normality"))
            if "homoscedasticity" in payload:
                homoscedasticity_payloads.append(payload.get("homoscedasticity"))

        if normality_payloads:
            counts["normality"] = _count_meets_assumption(normality_payloads)
        if homoscedasticity_payloads:
            counts["homoscedasticity"] = _count_meets_assumption(
                homoscedasticity_payloads
            )

    beta_fit = imputacion.get("beta_distribution_fit")
    if isinstance(beta_fit, dict):
        counts["beta_lambda_alto"] = {
            variable: _is_lambda_high(
                fit.get("severity_joint", {}).get("lambda")
                if isinstance(fit, dict)
                else None
            )
            for variable, fit in beta_fit.items()
        }

    severity = imputacion.get("severity")
    if isinstance(severity, dict) and "lambda" in severity:
        lambda_value = severity.get("lambda")
        if isinstance(lambda_value, dict):
            counts["lambda_global_alto"] = False
            counts["lambda_variables_alto"] = sum(
                1 for value in lambda_value.values() if _is_lambda_high(value)
            )
        else:
            counts["lambda_global_alto"] = _is_lambda_high(lambda_value)

    return counts


def _count_meets_assumption(payloads: Any) -> dict[str, int]:
    counts = {"cumple": 0, "no_cumple": 0, "no_aplica": 0}
    for payload in payloads:
        value = payload.get("meets_assumption") if isinstance(payload, dict) else None
        if value is True:
            counts["cumple"] += 1
        elif value is False:
            counts["no_cumple"] += 1
        else:
            counts["no_aplica"] += 1
    return counts


def _is_lambda_high(value: Any) -> bool:
    try:
        return float(value) > 0.30
    except (TypeError, ValueError):
        return False


def _try_parse_action(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    if parsed.get("accion") != "recalcular":
        return None
    parametros = parsed.get("parametros")
    if not isinstance(parametros, dict):
        return None
    return {"accion": "recalcular", "parametros": parametros}
