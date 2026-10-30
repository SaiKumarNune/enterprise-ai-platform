from app.services.llm import LLMTimeoutError, LLMUnavailableError


class ScriptedLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.index = 0
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            return "{}"
        if self.index < len(self.responses):
            text = self.responses[self.index]
            self.index += 1
            return text
        return self.responses[-1]


class UnavailableLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMUnavailableError("The local language model is unavailable")


class TimeoutLLM:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMTimeoutError("The local language model timed out")
