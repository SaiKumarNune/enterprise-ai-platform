"""Optional OpenAI-compatible hosted LLM for evaluation only.

No SDK is installed. Hosted mode never falls back to DatasetLLM or Ollama.
It does not run unless EVAL_HOSTED_LLM_BASE_URL and EVAL_HOSTED_LLM_MODEL are set.
An API key is optional so a local compatible proxy can be used without a paid account.
"""

from __future__ import annotations

import os

import httpx

from eval.live import LiveProbe, evaluation_database_url, probe_live_environment


class HostedLLMError(Exception):
    pass


class HostedOpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_seconds, connect=2.0)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise HostedLLMError("Hosted evaluation LLM is unavailable") from exc
        if response.status_code >= 400:
            raise HostedLLMError("Hosted evaluation LLM rejected the request")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise HostedLLMError("Hosted evaluation LLM returned an invalid payload") from exc
        if not isinstance(content, str) or not content.strip():
            raise HostedLLMError("Hosted evaluation LLM returned an empty response")
        return content.strip()


def hosted_config() -> tuple[str, str, str]:
    base = os.getenv("EVAL_HOSTED_LLM_BASE_URL", "").strip()
    model = os.getenv("EVAL_HOSTED_LLM_MODEL", "").strip()
    key = os.getenv("EVAL_HOSTED_LLM_API_KEY", "").strip()
    return base, model, key


def probe_hosted_environment() -> LiveProbe:
    base, model, _key = hosted_config()
    if not base or not model:
        return LiveProbe(
            False,
            "EVAL_HOSTED_LLM_BASE_URL and EVAL_HOSTED_LLM_MODEL are unset; "
            "hosted eval will not call a paid API",
        )
    live = probe_live_environment(require_llm=False)
    if not live.ready:
        return LiveProbe(
            False,
            f"Hosted generation still needs isolated retrieval: {live.reason}",
            database_url=live.database_url,
            database_name=live.database_name,
            embedder_id=live.embedder_id,
            llm_id=model,
        )
    return LiveProbe(
        True,
        "ok",
        database_url=live.database_url,
        database_name=live.database_name,
        embedder_id=live.embedder_id,
        llm_id=model,
    )


def build_hosted_client() -> HostedOpenAICompatibleClient:
    base, model, key = hosted_config()
    timeout = float(os.getenv("EVAL_HOSTED_LLM_TIMEOUT_SECONDS", "60"))
    return HostedOpenAICompatibleClient(base, model, key, timeout)


def hosted_not_run_reason() -> str:
    url = evaluation_database_url()
    if not url:
        return "Hosted mode needs EVAL_HOSTED_LLM_* plus an isolated *_test database URL"
    return "Hosted OpenAI-compatible evaluation is not configured"
