from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any

from dotenv import load_dotenv
import requests


DEFAULT_TIMEOUT_SECONDS = 30


class LLMRequestError(RuntimeError):
    """Error controlado al llamar a un proveedor LLM externo."""


class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = (
            f"{self.base_url}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": self._map_role(message.get("role", "user")),
                    "parts": [{"text": message.get("content", "")}],
                }
                for message in messages
            ]
        }

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
        try:
            response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise LLMRequestError("No fue posible llamar a Gemini.") from exc


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

    def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            raise LLMRequestError("No fue posible llamar a DeepSeek.") from exc
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

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

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
            raise LLMRequestError("No fue posible llamar a Ollama.") from exc
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
