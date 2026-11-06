"""MCP tool-boundary errors. Messages must not leak SQL, secrets, or identities."""


class McpToolError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
