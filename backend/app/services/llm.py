from __future__ import annotations

from typing import Protocol

import httpx

from app.core.config import settings


class LLMError(Exception):
    """Base class for local LLM failures."""


class LLMUnavailableError(LLMError):
    """Ollama is not reachable."""


class LLMTimeoutError(LLMError):
    """The local LLM did not respond in time."""


class LLMBackend(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a completion for the given system and user messages."""


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds, connect=2.0)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.ConnectError as exc:
            raise LLMUnavailableError("The local language model is unavailable") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("The local language model timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailableError("The local language model is unavailable") from exc

        if response.status_code >= 500:
            raise LLMUnavailableError("The local language model is unavailable")
        if response.status_code >= 400:
            raise LLMError("The local language model rejected the request")

        try:
            body = response.json()
            content = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("The local language model returned an invalid response") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("The local language model returned an empty response")
        return content.strip()


_llm: OllamaChatClient | None = None


def get_llm() -> LLMBackend:
    global _llm
    if _llm is None:
        _llm = OllamaChatClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    return _llm
