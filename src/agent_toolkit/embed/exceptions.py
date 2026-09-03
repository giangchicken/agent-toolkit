"""shape. The error taxonomy this route raises, its own and not the chat route's."""


class EmbedError(Exception):
    def __init__(
        self,
        message: str,
        details: dict[str, object] | None = None,
        provider: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.provider = provider

    def __str__(self) -> str:
        provider_prefix = f"[{self.provider}] " if self.provider else ""
        if self.details:
            return f"{provider_prefix}{self.message} (details: {self.details})"
        return f"{provider_prefix}{self.message}"


class EmbedConfigError(EmbedError):
    pass


class EmbedProviderError(EmbedError):
    pass


class EmbedAPIError(EmbedError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        provider: str | None = None,
        details: dict[str, object] | None = None,
    ):
        super().__init__(message, details, provider)
        self.status_code = status_code

    def __str__(self) -> str:
        parts = []
        if self.provider:
            parts.append(f"[{self.provider}]")
        if self.status_code:
            parts.append(f"HTTP {self.status_code}")
        parts.append(self.message)
        return " ".join(parts)


class EmbedTimeoutError(EmbedAPIError):
    def __init__(
        self,
        message: str = "Request timed out",
        timeout: float | None = None,
        provider: str | None = None,
    ):
        super().__init__(message, status_code=408, provider=provider)
        self.timeout = timeout


class EmbedRateLimitError(EmbedAPIError):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        provider: str | None = None,
    ):
        super().__init__(message, status_code=429, provider=provider)
        self.retry_after = retry_after


class EmbedAuthenticationError(EmbedAPIError):
    def __init__(
        self,
        message: str = "Authentication failed",
        provider: str | None = None,
    ):
        super().__init__(message, status_code=401, provider=provider)


class EmbedModelNotFoundError(EmbedAPIError):
    def __init__(
        self,
        message: str = "Model not found",
        model: str | None = None,
        provider: str | None = None,
    ):
        super().__init__(message, status_code=404, provider=provider)
        self.model = model


class EmbedInputTooLargeError(EmbedAPIError):
    """One input over the model's token limit, not a conversation over a window.

    The chat route's ``ProviderContextWindowError`` counts a whole request; here
    the limit is per text, so the caller's remedy is to chunk that text rather
    than to drop earlier turns.
    """


class EmbedQuotaExceededError(EmbedRateLimitError):
    pass


__all__ = [
    "EmbedAPIError",
    "EmbedAuthenticationError",
    "EmbedConfigError",
    "EmbedError",
    "EmbedInputTooLargeError",
    "EmbedModelNotFoundError",
    "EmbedProviderError",
    "EmbedQuotaExceededError",
    "EmbedRateLimitError",
    "EmbedTimeoutError",
]
