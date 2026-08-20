from __future__ import annotations

from typing import Any

import pandas as pd

from llm.client import LLMClient
from llm.explainer import PipelineExplainer


class FakeLLMClient(LLMClient):
    def __init__(self, response: str = "explicacion fija") -> None:
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


def make_pipeline_result() -> dict[str, Any]:
    return {
        "decision": "mice",
        "reasoning": "max_pct=10.00, ratio=12.00, goal=inference",
        "diagnostico": {"detalle": "no debe ser enviado"},
        "imputed_data": pd.DataFrame(
            {
                "x": ["DATO_SECRETO_X", "otro_valor"],
                "y": [1.0, 2.0],
            }
        ),
        "imputer_report": {
            "severity": {
                "lambda": {
                    "x": 0.2,
                }
            }
        },
        "warnings": ["advertencia agregada"],
    }


def messages_as_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def test_explain_never_sends_imputed_data_or_dataframe_contents() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_pipeline_result(), client=client)

    result = explainer.explain()

    sent_text = messages_as_text(client.calls[0])
    assert result == "explicacion fija"
    assert "imputed_data" not in sent_text
    assert "DATO_SECRETO_X" not in sent_text
    assert "otro_valor" not in sent_text
    assert "diagnostico" not in sent_text
    assert "mice" in sent_text
    assert "advertencia agregada" in sent_text
    assert "No te limites a reportar los numeros" in sent_text
    assert client.generation_options_calls[0] == {
        "temperature": 0.2,
        "num_predict": 400,
    }


def test_ask_without_previous_explain_builds_base_context_first() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_pipeline_result(), client=client)

    response = explainer.ask("Que significa lambda?")

    assert response == "explicacion fija"
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
        {"temperature": 0.2, "num_predict": 400},
        {"temperature": 0.4, "num_predict": 350},
    ]


def test_history_accumulates_between_successive_ask_calls() -> None:
    client = FakeLLMClient()
    explainer = PipelineExplainer(make_pipeline_result(), client=client)

    first = explainer.ask("Primera pregunta")
    second = explainer.ask("Segunda pregunta")

    assert first == "explicacion fija"
    assert second == "explicacion fija"
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
    explainer = PipelineExplainer(make_pipeline_result(), client=client)
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
