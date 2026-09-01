from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import llm.explainer as explainer_module
from llm.client import LLMClient
from llm.explainer import (
    PipelineExplainer,
    _compute_assumption_counts,
    _sanitize_for_llm,
    _try_parse_action,
)


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        response: str | None = None,
    ) -> None:
        self.responses = [response] if response is not None else []
        self.calls: list[list[dict[str, str]]] = []
        self.generation_options_calls: list[dict[str, Any] | None] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append([message.copy() for message in messages])
        self.generation_options_calls.append(generation_options)
        if not self.responses:
            section = messages[-1]["content"].splitlines()[0].replace("Seccion: ", "")
            return f"respuesta {section}"
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def make_report() -> dict[str, Any]:
    return {
        "metadata": {"dataset_name": "demo", "n_rows": 2, "n_columns": 2},
        "exploracion_previa": {
            "plots_generated": [
                str(Path("outputs") / "demo" / "plots" / "gg_miss_var.png"),
                str(Path("outputs") / "demo" / "plots" / "aggr.png"),
            ],
            "comparison_plots": {
                "x": {
                    "boxplot": "<div>HTML_GIGANTE_PLOTLY</div>",
                }
            },
            "missing_data_status": {"mcar": False},
        },
        "decision": {
            "metodo": "mice",
            "razonamiento": "max_pct=10.00, ratio=12.00, goal=inference",
            "criterios": {"max_pct": 10.0, "ratio": 12.0, "goal": "inference"},
        },
        "resumen_descriptivo": {
            "x": {
                "antes": {"media": 1.0},
                "despues": {"media": 1.5},
            }
        },
        "imputacion": {
            "severity": {"lambda": {"x": 0.2}},
            "assumptions": {
                "convergence": {
                    "x": {
                        "converged": True,
                        "chain_mean": [[1.0, 1.1], [1.2, 1.3]],
                    }
                },
                "distribution_comparison": {
                    "x": {
                        "ad_statistic": 2.5,
                        "p_value": 0.04,
                        "meets_assumption": False,
                    }
                },
            },
            "beta_distribution_fit": {
                "x": {
                    "shape1_pooled": 2.0,
                    "shape2_pooled": 3.0,
                    "severity_joint": {"lambda": 0.2},
                    "severity_per_parameter": {
                        "shape1": {"lambda": 0.1},
                    },
                }
            },
        },
        "advertencias": ["advertencia agregada"],
    }


def messages_as_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def test_explain_sends_sanitized_full_report_without_dataframe_contents() -> None:
    client = FakeLLMClient()
    report = make_report()
    report["imputed_data"] = pd.DataFrame(
        {
            "x": ["DATO_SECRETO_X", "otro_valor"],
            "y": [1.0, 2.0],
        }
    )
    explainer = PipelineExplainer(report, client=client)

    result = explainer.explain()

    sent_text = "\n".join(messages_as_text(call) for call in client.calls)
    assert result == {
        "general": "respuesta general",
        "exploracion": "respuesta exploracion",
        "resumen_descriptivo": "respuesta resumen_descriptivo",
        "decision": "respuesta decision",
        "imputacion": "respuesta imputacion",
        "supuestos": "respuesta supuestos",
        "advertencias_y_sugerencias": "respuesta advertencias_y_sugerencias",
    }
    assert explainer.last_structured_explanation == result
    assert len(client.calls) == 7
    assert "imputed_data" not in sent_text
    assert "DATO_SECRETO_X" not in sent_text
    assert "otro_valor" not in sent_text
    assert "HTML_GIGANTE_PLOTLY" not in sent_text
    assert "gg_miss_var.png" in sent_text
    assert "chain_mean" not in sent_text
    assert "detalle_convergence" in sent_text
    assert "mice" in sent_text
    assert "advertencia agregada" in sent_text
    assert "UNA PARTE especifica" in sent_text
    assert "REGLA ABSOLUTA" in sent_text
    assert "no tienes acceso a ningun archivo" in sent_text
    assert all(
        options == {"temperature": 0.1, "num_predict": 400, "num_ctx": 4096}
        for options in client.generation_options_calls
    )


def test_explain_sends_only_section_specific_payloads() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_report(), client=client)

    explainer.explain()

    calls_by_section = {
        call[1]["content"].splitlines()[0].replace("Seccion: ", ""): call
        for call in client.calls
    }
    general_payload = calls_by_section["general"][1]["content"]
    decision_payload = calls_by_section["decision"][1]["content"]
    imputacion_payload = calls_by_section["imputacion"][1]["content"]
    supuestos_payload = calls_by_section["supuestos"][1]["content"]

    assert "resumen_supuestos" in general_payload
    assert "advertencias_automaticas_count" in general_payload
    assert "advertencias_count" not in general_payload
    assert "razonamiento" in decision_payload
    assert "assumptions" not in decision_payload
    assert "exploracion_previa" not in decision_payload
    assert "point_estimate" not in decision_payload
    assert "assumptions" not in imputacion_payload
    assert "covariance_estimate" not in imputacion_payload
    assert "severity_per_parameter" not in imputacion_payload
    assert "detalle_convergence" in supuestos_payload
    assert "ad_statistic" not in supuestos_payload


def test_explain_computes_assumption_counts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = explainer_module._compute_assumption_counts
    calls = {"count": 0}

    def spy(imputacion: dict[str, Any] | None) -> dict[str, Any]:
        calls["count"] += 1
        return original(imputacion)

    monkeypatch.setattr(explainer_module, "_compute_assumption_counts", spy)

    PipelineExplainer(make_report(), client=FakeLLMClient()).explain()

    assert calls["count"] == 1


def test_explain_uses_fallback_when_imputation_response_is_empty() -> None:
    client = FakeLLMClient()
    client.responses = [
        "general ok",
        "exploracion ok",
        "resumen ok",
        "decision ok",
        "   ",
        "supuestos ok",
        "advertencias ok",
    ]
    explainer = PipelineExplainer(make_report(), client=client)

    result = explainer.explain()

    assert "interpretacion de IA" in result["imputacion"]
    assert "metricas de severidad" in result["imputacion"]
    assert "ajuste de distribucion Beta" in result["imputacion"]
    assert result["supuestos"] == "supuestos ok"


def test_sanitize_for_llm_removes_comparison_plots_and_keeps_plot_names_only() -> None:
    sanitized = _sanitize_for_llm(make_report())

    exploration = sanitized["exploracion_previa"]
    assert "comparison_plots" not in exploration
    assert exploration["plots_generated"] == ["gg_miss_var.png", "aggr.png"]
    assert exploration["missing_data_status"] == {"mcar": False}
    convergence = sanitized["imputacion"]["assumptions"]["convergence"]["x"]
    assert convergence == {"converged": True}


def test_compute_assumption_counts_summarizes_statuses_and_lambda() -> None:
    imputacion = {
        "severity": {"lambda": {"x": 0.31, "y": 0.2, "z": None}},
        "assumptions": {
            "convergence": {
                "x": {"converged": True},
                "y": {"converged": False},
                "z": {},
            },
            "distribution_comparison": {
                "x": {"meets_assumption": True},
                "y": {"meets_assumption": False},
                "z": {},
            },
            "x": {
                "normality": {"meets_assumption": True},
                "homoscedasticity": {"meets_assumption": False},
            },
            "y": {
                "normality": {},
                "homoscedasticity": {"meets_assumption": True},
            },
        },
        "beta_distribution_fit": {
            "Ind_Mobile": {"severity_joint": {"lambda": 0.29}},
            "Other": {"severity_joint": {"lambda": 0.4}},
        },
    }

    counts = _compute_assumption_counts(imputacion)

    assert counts == {
        "convergence": {"convergio": 1, "no_convergio": 1, "no_aplica": 1},
        "distribution_comparison": {"cumple": 1, "no_cumple": 1, "no_aplica": 1},
        "normality": {"cumple": 1, "no_cumple": 0, "no_aplica": 1},
        "homoscedasticity": {"cumple": 1, "no_cumple": 1, "no_aplica": 0},
        "beta_lambda_alto": {"Ind_Mobile": False, "Other": True},
        "lambda_global_alto": False,
        "lambda_variables_alto": 1,
    }


def test_compute_assumption_counts_handles_scalar_lambda() -> None:
    counts = _compute_assumption_counts({"severity": {"lambda": 0.35}})

    assert counts["lambda_global_alto"] is True


def test_try_parse_action_detects_valid_recalculation_action() -> None:
    action = _try_parse_action(
        '{"accion": "recalcular", "parametros": {"m": 10, "goal": "inference"}}'
    )

    assert action == {
        "accion": "recalcular",
        "parametros": {"m": 10, "goal": "inference"},
    }
    assert _try_parse_action("explicacion normal") is None
    assert _try_parse_action('{"accion": "otro", "parametros": {}}') is None


def test_ask_without_previous_explain_builds_base_context_first() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_report(), client=client)

    response = explainer.ask("Que significa lambda?")

    assert response == "respuesta Que significa lambda?"
    assert len(client.calls) == 8
    assert [message["role"] for message in client.calls[0]] == ["system", "user"]
    assert [message["role"] for message in client.calls[7]] == [
        "system",
        "assistant",
        "user",
    ]
    assert "## General" in explainer.history[1]["content"]
    assert "## Advertencias Y Sugerencias" in explainer.history[1]["content"]
    assert client.calls[7][-1]["content"] == "Que significa lambda?"
    assert client.generation_options_calls[:7] == [
        {"temperature": 0.1, "num_predict": 400, "num_ctx": 4096}
    ] * 7
    assert client.generation_options_calls[7] == {
        "temperature": 0.15,
        "num_predict": 700,
        "num_ctx": 8192,
    }


def test_ask_executes_tool_when_model_requests_recalculation() -> None:
    client = FakeLLMClient()
    client.responses = [
        '{"accion": "recalcular", "parametros": {"m": 10, "beta_vars": ["x"]}}',
        "Resultado recalculado explicado.",
    ]
    seen: dict[str, Any] = {}

    def tool_executor(parametros: dict[str, Any]) -> dict[str, Any]:
        seen["parametros"] = parametros
        return {
            "decision": "mice",
            "reasoning": "recalculado",
            "warnings": [],
            "imputer_report": {
                "severity": {"lambda": 0.1},
                "point_estimate": {"x": 1.0},
                "beta_distribution_fit": {"x": {"shape1_pooled": 2.0}},
            },
            "imputed_data": pd.DataFrame({"x": ["NO_ENVIAR"]}),
        }

    explainer = PipelineExplainer(
        make_report(),
        client=client,
        tool_executor=tool_executor,
    )
    explainer.history = [
        {"role": "system", "content": "system inicial"},
        {"role": "user", "content": "resumen inicial"},
        {"role": "assistant", "content": "explicacion inicial"},
    ]
    response = explainer.ask("prueba con mas imputaciones y beta en x")

    assert response == "Resultado recalculado explicado."
    assert seen["parametros"] == {"m": 10, "beta_vars": ["x"]}
    assert len(client.calls) == 2
    sent_text = messages_as_text(client.calls[1])
    assert "Resultado del recalculo solicitado" in sent_text
    assert "recalculado" in sent_text
    assert "NO_ENVIAR" not in sent_text
    assert client.generation_options_calls[1] == {
        "temperature": 0.15,
        "num_predict": 1000,
        "num_ctx": 8192,
    }


def test_ask_keeps_role_alternation_after_recalculation_action() -> None:
    client = FakeLLMClient()
    action_json = '{"accion": "recalcular", "parametros": {"m": 10}}'
    client.responses = [action_json, "Explicacion final del recalculo."]

    def tool_executor(parametros: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": "mice",
            "reasoning": "recalculado",
            "warnings": [],
            "imputer_report": {"severity": {"lambda": 0.1}},
        }

    explainer = PipelineExplainer(
        make_report(),
        client=client,
        tool_executor=tool_executor,
    )
    explainer.history = [
        {"role": "system", "content": "system inicial"},
        {"role": "assistant", "content": "explicacion inicial"},
    ]

    response = explainer.ask("recalcula con m=10")

    assert response == "Explicacion final del recalculo."
    assert [message["role"] for message in explainer.history[-4:]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert explainer.history[-3]["content"] == action_json
    assert all(
        before["role"] != after["role"]
        for before, after in zip(explainer.history[-4:], explainer.history[-3:])
    )


def test_ask_returns_fallback_when_model_response_is_empty() -> None:
    client = FakeLLMClient("   ")
    explainer = PipelineExplainer(make_report(), client=client)
    explainer.history = [
        {"role": "system", "content": "system inicial"},
        {"role": "assistant", "content": "explicacion inicial"},
    ]

    response = explainer.ask("pregunta normal")

    assert response == (
        "No pude generar una explicacion del recalculo, pero la operacion se "
        "ejecuto correctamente. Revisa el reporte actualizado o intenta "
        "reformular tu pregunta."
    )
    assert explainer.history[-1]["content"] == response


def test_ask_returns_fallback_when_recalculation_explanation_is_empty() -> None:
    client = FakeLLMClient()
    client.responses = ['{"accion": "recalcular", "parametros": {"m": 10}}', ""]

    def tool_executor(parametros: dict[str, Any]) -> dict[str, Any]:
        return {
            "decision": "mice",
            "reasoning": "recalculado",
            "warnings": [],
            "imputer_report": {"severity": {"lambda": 0.1}},
        }

    explainer = PipelineExplainer(
        make_report(),
        client=client,
        tool_executor=tool_executor,
    )
    explainer.history = [
        {"role": "system", "content": "system inicial"},
        {"role": "assistant", "content": "explicacion inicial"},
    ]

    response = explainer.ask("recalcula con m=10")

    assert response == (
        "No pude generar una explicacion del recalculo, pero la operacion se "
        "ejecuto correctamente. Revisa el reporte actualizado o intenta "
        "reformular tu pregunta."
    )


def test_ask_reports_tool_unavailable_when_no_executor_is_configured() -> None:
    client = FakeLLMClient(
        '{"accion": "recalcular", "parametros": {"m": 10}}'
    )
    explainer = PipelineExplainer(make_report(), client=client)
    explainer.history = [
        {"role": "system", "content": "system inicial"},
        {"role": "user", "content": "resumen inicial"},
        {"role": "assistant", "content": "explicacion inicial"},
    ]

    response = explainer.ask("prueba con mas imputaciones")

    assert "no esta disponible" in response


def test_history_accumulates_between_successive_ask_calls() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_report(), client=client)

    first = explainer.ask("Primera pregunta")
    second = explainer.ask("Segunda pregunta")

    assert first == "respuesta Primera pregunta"
    assert second == "respuesta Segunda pregunta"
    assert [message["role"] for message in explainer.history] == [
        "system",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert explainer.history[-4]["content"] == "Primera pregunta"
    assert explainer.history[-2]["content"] == "Segunda pregunta"
    assert len(client.calls) == 9
    assert len(client.calls[-1]) == 5


def test_ask_trims_long_history_before_calling_client() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_report(), client=client)
    explainer.history = [
        {"role": "system", "content": "system inicial"},
        {"role": "user", "content": "resumen inicial"},
    ]
    for index in range(11):
        explainer.history.append({"role": "assistant", "content": f"mensaje {index}"})

    explainer.ask("Pregunta final")

    sent_messages = client.calls[0]
    assert len(sent_messages) == 11
    assert sent_messages[0]["content"].startswith("system inicial")
    assert "capacidad de EJECUTAR nuevos analisis" in sent_messages[0]["content"]
    assert "excluir_variables" in sent_messages[0]["content"]
    assert "variabilidad aleatoria normal entre corridas" in sent_messages[0]["content"]
    assert "NUNCA afirmes de forma categorica" in sent_messages[0]["content"]
    assert sent_messages[1]["content"] == "resumen inicial"
    assert sent_messages[2]["content"] == "mensaje 3"
    assert sent_messages[-1]["content"] == "Pregunta final"


def test_domain_context_is_only_included_when_not_empty() -> None:
    client_without_context = FakeLLMClient()
    PipelineExplainer(
        make_report(),
        client=client_without_context,
        domain_context=None,
    ).explain()

    system_message = client_without_context.calls[0][0]["content"]
    assert "Contexto del dataset:" not in system_message

    client_with_context = FakeLLMClient()
    PipelineExplainer(
        make_report(),
        client=client_with_context,
        domain_context="Encuesta mundial de felicidad 2023.",
    ).explain()

    system_message = client_with_context.calls[0][0]["content"]
    assert "Contexto del dataset: Encuesta mundial de felicidad 2023." in system_message


def test_advertencias_prompt_limits_suggested_actions_to_executable_types() -> None:
    explainer = PipelineExplainer(make_report(), client=FakeLLMClient())

    prompt = explainer._system_prompt_seccion("advertencias_y_sugerencias")

    assert "unicamente de estos 4 tipos" in prompt
    assert "Aumentar el numero de imputaciones" in prompt
    assert "Cambiar el objetivo del analisis" in prompt
    assert "Agregar ajuste de distribucion Beta" in prompt
    assert "Excluir una variable especifica" in prompt
    assert "deja acciones_sugeridas como una lista vacia" in prompt
