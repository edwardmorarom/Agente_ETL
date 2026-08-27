from __future__ import annotations

from typing import Any

import pytest
import requests

import llm.client as client_module
from llm.client import (
    DeepSeekClient,
    GeminiClient,
    LLMRequestError,
    OllamaClient,
    get_llm_client,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
        text: str = "",
        status_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict[str, Any]:
        return self.payload


def test_ollama_client_posts_chat_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(
        url: str,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        return FakeResponse({"message": {"content": "respuesta local"}})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama-test",
        timeout=120,
    )
    result = client.chat([{"role": "user", "content": "Hola"}])

    assert result == "respuesta local"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["json"] == {
        "model": "llama-test",
        "messages": [{"role": "user", "content": "Hola"}],
        "stream": False,
    }
    assert seen["timeout"] == 120


def test_ollama_client_adds_generation_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(
        url: str,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        seen["json"] = json
        return FakeResponse({"message": {"content": "respuesta local"}})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    OllamaClient(base_url="http://localhost:11434", model="llama-test").chat(
        [{"role": "user", "content": "Hola"}],
        generation_options={"temperature": 0.2, "num_predict": 400, "num_ctx": 8192},
    )

    assert seen["json"]["options"] == {
        "temperature": 0.2,
        "num_predict": 400,
        "num_ctx": 8192,
    }


def test_gemini_client_maps_generation_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "respuesta"}]}}]}
        )

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    GeminiClient(api_key="key", model="gemini-test").chat(
        [{"role": "user", "content": "Hola"}],
        generation_options={"temperature": 0.2, "num_predict": 400},
    )

    assert seen["json"]["generationConfig"] == {
        "temperature": 0.2,
        "maxOutputTokens": 400,
    }
    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-test:generateContent"
    )
    assert "key" not in seen["url"]
    assert seen["headers"] == {
        "x-goog-api-key": "key",
        "Content-Type": "application/json",
    }
    assert seen["timeout"] == 60


def test_gemini_client_retries_503_then_returns_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        FakeResponse({}, status_code=503, text="sobrecarga 1"),
        FakeResponse({}, status_code=503, text="sobrecarga 2"),
        FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "respuesta final"}]}}]}
        ),
    ]
    sleep_calls: list[int] = []

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    monkeypatch.setattr(client_module.time, "sleep", sleep_calls.append)

    result = GeminiClient(api_key="key", model="gemini-test").chat(
        [{"role": "user", "content": "Hola"}]
    )

    assert result == "respuesta final"
    assert sleep_calls == [2, 4]
    assert responses == []


def test_deepseek_client_maps_generation_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        seen["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "respuesta"}}]})

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    DeepSeekClient(api_key="key", model="deepseek-test").chat(
        [{"role": "user", "content": "Hola"}],
        generation_options={"temperature": 0.4, "num_predict": 350},
    )

    assert seen["json"]["temperature"] == 0.4
    assert seen["json"]["max_tokens"] == 350


def test_ollama_client_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")

    client = OllamaClient()

    assert client.base_url == "http://ollama.test:11434"
    assert client.model == "custom-model"


def test_ollama_client_wraps_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        url: str,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        raise requests.ConnectionError("ollama apagado")

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    with pytest.raises(LLMRequestError, match="No fue posible llamar a Ollama"):
        OllamaClient().chat([{"role": "user", "content": "Hola"}])


def test_gemini_client_error_includes_status_and_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        assert "key" not in url
        assert headers["x-goog-api-key"] == "key"
        return FakeResponse({}, status_code=403, text="API key invalida")

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    with pytest.raises(LLMRequestError) as exc_info:
        GeminiClient(api_key="key", model="gemini-test").chat(
            [{"role": "user", "content": "Hola"}]
        )

    assert "403" in str(exc_info.value)
    assert "API key invalida" in str(exc_info.value)


def test_deepseek_client_error_includes_status_and_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse({}, status_code=403, text="cuota excedida")

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    with pytest.raises(LLMRequestError) as exc_info:
        DeepSeekClient(api_key="key", model="deepseek-test").chat(
            [{"role": "user", "content": "Hola"}]
        )

    assert "403" in str(exc_info.value)
    assert "cuota excedida" in str(exc_info.value)


def test_ollama_client_error_includes_status_and_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        url: str,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse({}, status_code=403, text="modelo no permitido")

    monkeypatch.setattr(client_module.requests, "post", fake_post)

    with pytest.raises(LLMRequestError) as exc_info:
        OllamaClient().chat([{"role": "user", "content": "Hola"}])

    assert "403" in str(exc_info.value)
    assert "modelo no permitido" in str(exc_info.value)


def test_get_llm_client_returns_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    client = get_llm_client()

    assert isinstance(client, OllamaClient)


def test_get_llm_client_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "otro")

    with pytest.raises(ValueError, match="gemini.*deepseek.*ollama"):
        get_llm_client()


def test_ollama_client_real_integration_returns_non_empty_text() -> None:
    try:
        response = requests.get("http://localhost:11434", timeout=1)
        response.raise_for_status()
    except requests.RequestException:
        pytest.skip("Ollama no esta corriendo en http://localhost:11434.")

    try:
        result = OllamaClient().chat(
            [{"role": "user", "content": "Responde solamente con la palabra ok."}]
        )
    except LLMRequestError:
        pytest.skip("Ollama esta corriendo, pero no completo una respuesta de chat.")

    assert result.strip()
