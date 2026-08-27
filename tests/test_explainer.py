from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from llm.client import LLMClient
from llm.explainer import (
    PipelineExplainer,
    _parse_structured_response,
    _sanitize_for_llm,
)


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        response: str = (
            '{"general":"general ok","exploracion":"exploracion ok",'
            '"resumen_descriptivo":"resumen ok","decision":"decision ok",'
            '"imputacion":"imputacion ok","supuestos":"supuestos ok",'
            '"advertencias":"advertencias ok"}'
        ),
    ) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []
        self.generation_options_calls: list[dict[str, Any] | None] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append([message.copy() for message in messages])
        self.generation_options_calls.append(generation_options)
        return self.response


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
            },
            "beta_distribution_fit": {
                "x": {
                    "shape1_pooled": 2.0,
                    "shape2_pooled": 3.0,
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

    sent_text = messages_as_text(client.calls[0])
    assert result == {
        "general": "general ok",
        "exploracion": "exploracion ok",
        "resumen_descriptivo": "resumen ok",
        "decision": "decision ok",
        "imputacion": "imputacion ok",
        "supuestos": "supuestos ok",
        "advertencias": "advertencias ok",
    }
    assert explainer.last_structured_explanation == result
    assert "imputed_data" not in sent_text
    assert "DATO_SECRETO_X" not in sent_text
    assert "otro_valor" not in sent_text
    assert "HTML_GIGANTE_PLOTLY" not in sent_text
    assert "gg_miss_var.png" in sent_text
    assert "chain_mean" not in sent_text
    assert "converged" in sent_text
    assert "mice" in sent_text
    assert "advertencia agregada" in sent_text
    assert "resultado completo de un pipeline" in sent_text
    assert "REGLA ABSOLUTA" in sent_text
    assert "no tienes acceso a ningun archivo" in sent_text
    assert client.generation_options_calls[0] == {
        "temperature": 0.1,
        "num_predict": 1200,
        "num_ctx": 8192,
    }


def test_sanitize_for_llm_removes_comparison_plots_and_keeps_plot_names_only() -> None:
    sanitized = _sanitize_for_llm(make_report())

    exploration = sanitized["exploracion_previa"]
    assert "comparison_plots" not in exploration
    assert exploration["plots_generated"] == ["gg_miss_var.png", "aggr.png"]
    assert exploration["missing_data_status"] == {"mcar": False}
    convergence = sanitized["imputacion"]["assumptions"]["convergence"]["x"]
    assert convergence == {"converged": True}


def test_parse_structured_response_falls_back_to_general_for_invalid_json() -> None:
    assert _parse_structured_response("texto libre") == {"general": "texto libre"}


def test_ask_without_previous_explain_builds_base_context_first() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_report(), client=client)

    response = explainer.ask("Que significa lambda?")

    assert response == (
        '{"general":"general ok","exploracion":"exploracion ok",'
        '"resumen_descriptivo":"resumen ok","decision":"decision ok",'
        '"imputacion":"imputacion ok","supuestos":"supuestos ok",'
        '"advertencias":"advertencias ok"}'
    )
    assert len(client.calls) == 2
    assert [message["role"] for message in client.calls[0]] == ["system", "user"]
    assert [message["role"] for message in client.calls[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert client.calls[1][-1]["content"] == "Que significa lambda?"
    assert client.generation_options_calls == [
        {"temperature": 0.1, "num_predict": 1200, "num_ctx": 8192},
        {"temperature": 0.15, "num_predict": 700, "num_ctx": 8192},
    ]


def test_history_accumulates_between_successive_ask_calls() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_report(), client=client)

    first = explainer.ask("Primera pregunta")
    second = explainer.ask("Segunda pregunta")

    assert first == (
        '{"general":"general ok","exploracion":"exploracion ok",'
        '"resumen_descriptivo":"resumen ok","decision":"decision ok",'
        '"imputacion":"imputacion ok","supuestos":"supuestos ok",'
        '"advertencias":"advertencias ok"}'
    )
    assert second == first
    assert [message["role"] for message in explainer.history] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert explainer.history[-4]["content"] == "Primera pregunta"
    assert explainer.history[-2]["content"] == "Segunda pregunta"
    assert len(client.calls) == 3
    assert len(client.calls[-1]) == 6


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
    assert sent_messages[0]["content"] == "system inicial"
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

    user_message = client_without_context.calls[0][1]["content"]
    assert not user_message.startswith(
        "Contexto del dataset proporcionado por el usuario:"
    )

    client_with_context = FakeLLMClient()
    PipelineExplainer(
        make_report(),
        client=client_with_context,
        domain_context="Encuesta mundial de felicidad 2023.",
    ).explain()

    user_message = client_with_context.calls[0][1]["content"]
    assert user_message.splitlines()[0] == (
        "Contexto del dataset proporcionado por el usuario: "
        "Encuesta mundial de felicidad 2023."
    )
