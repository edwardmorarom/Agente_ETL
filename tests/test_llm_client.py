from __future__ import annotations

from typing import Any

import pytest
import requests

import llm.client as client_module
from llm.client import LLMRequestError, OllamaClient, get_llm_client


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_error: Exception | None = None) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

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

    client = OllamaClient(base_url="http://localhost:11434", model="llama-test")
    result = client.chat([{"role": "user", "content": "Hola"}])

    assert result == "respuesta local"
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["json"] == {
        "model": "llama-test",
        "messages": [{"role": "user", "content": "Hola"}],
        "stream": False,
    }
    assert seen["timeout"] == 120


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
