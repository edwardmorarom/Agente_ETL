from __future__ import annotations

from abc import ABC, abstractmethod
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
import requests


DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = [2, 4, 8]
RETRYABLE_STATUS_CODES = {429, 503}


class LLMRequestError(RuntimeError):
    """Error controlado al llamar a un proveedor LLM externo."""


def _request_error_message(provider: str, exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        return (
            f"No fue posible llamar a {provider}. Status: {response.status_code}. "
            f"Detalle: {response.text[:500]}"
        )
    return f"No fue posible llamar a {provider}. Error de conexion: {exc}"


def _status_code_from_exception(exc: requests.RequestException) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    return getattr(response, "status_code", None)


def _request_with_retries(
    provider: str,
    send_request: Callable[[], requests.Response],
) -> requests.Response:
    last_exc: requests.RequestException | None = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            response = send_request()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            status_code = _status_code_from_exception(exc)
            should_retry = (
                status_code in RETRYABLE_STATUS_CODES
                and attempt < MAX_RETRY_ATTEMPTS - 1
            )
            if not should_retry:
                raise LLMRequestError(_request_error_message(provider, exc)) from exc
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    if last_exc is not None:
        raise LLMRequestError(_request_error_message(provider, last_exc))
    raise LLMRequestError(f"No fue posible llamar a {provider}.")


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        ...


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.timeout = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "60"))

    def chat(
        self,
        messages: list[dict[str, str]],
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": self._map_role(message.get("role", "user")),
                    "parts": [{"text": message.get("content", "")}],
                }
                for message in messages
            ]
        }
        if generation_options is not None:
            generation_config: dict[str, Any] = {}
            if "temperature" in generation_options:
                generation_config["temperature"] = generation_options["temperature"]
            if "num_predict" in generation_options:
                generation_config["maxOutputTokens"] = generation_options["num_predict"]
            if generation_config:
                payload["generationConfig"] = generation_config

        data = self._post(url, payload)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("Respuesta invalida de Gemini.") from exc

    def _map_role(self, role: str) -> str:
        if role == "assistant":
            return "model"
        return "user"

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        response = _request_with_retries(
            "Gemini",
            lambda: requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            ),
        )
        return response.json()


class DeepSeekClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/chat/completions",
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if generation_options is not None:
            if "temperature" in generation_options:
                payload["temperature"] = generation_options["temperature"]
            if "num_predict" in generation_options:
                payload["max_tokens"] = generation_options["num_predict"]

        try:
            response = _request_with_retries(
                "DeepSeek",
                lambda: requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                ),
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("Respuesta invalida de DeepSeek.") from exc


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        load_dotenv()
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout = timeout or int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

    def chat(
        self,
        messages: list[dict[str, str]],
        generation_options: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if generation_options is not None:
            payload["options"] = generation_options

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.RequestException as exc:
            raise LLMRequestError(_request_error_message("Ollama", exc)) from exc
        except (KeyError, TypeError) as exc:
            raise LLMRequestError("Respuesta invalida de Ollama.") from exc


def get_llm_client() -> LLMClient:
    load_dotenv()
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        return GeminiClient()
    if provider == "deepseek":
        return DeepSeekClient()
    if provider == "ollama":
        return OllamaClient()

    raise ValueError(
        "LLM_PROVIDER debe ser uno de: 'gemini', 'deepseek', 'ollama'."
    )
