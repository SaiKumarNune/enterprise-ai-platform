from app.services.llm import LLMTimeoutError, LLMUnavailableError


class ScriptedLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class UnavailableLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMUnavailableError("The local language model is unavailable")


class TimeoutLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMTimeoutError("The local language model timed out")
