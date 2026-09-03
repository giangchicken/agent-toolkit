"""logic. Which provider failure this is, and whether to retry it."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp
import openai

from agent_toolkit.embed.exceptions import (
    EmbedAPIError,
    EmbedAuthenticationError,
    EmbedConfigError,
    EmbedError,
    EmbedInputTooLargeError,
    EmbedRateLimitError,
    EmbedTimeoutError,
)

__all__ = ["is_retriable", "map_error"]

ErrorClassifier = Callable[[Exception], bool]


@dataclass(frozen=True)
class MappingRule:
    classifier: ErrorClassifier
    factory: Callable[[Exception, str | None], EmbedError]


def _instance_of(*types: type[BaseException]) -> ErrorClassifier:
    return lambda exc: isinstance(exc, types)


def _message_contains(*needles: str) -> ErrorClassifier:
    def _classifier(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(needle in msg for needle in needles)

    return _classifier


_GLOBAL_RULES: list[MappingRule] = [
    MappingRule(
        classifier=_instance_of(openai.AuthenticationError),
        factory=lambda exc, provider: EmbedAuthenticationError(
            str(exc), provider=provider
        ),
    ),
    MappingRule(
        classifier=_instance_of(openai.RateLimitError),
        factory=lambda exc, provider: EmbedRateLimitError(str(exc), provider=provider),
    ),
    MappingRule(
        classifier=_message_contains("rate limit", "429", "quota"),
        factory=lambda exc, provider: EmbedRateLimitError(str(exc), provider=provider),
    ),
    MappingRule(
        classifier=_message_contains(
            "context length", "maximum context", "too long", "input is too large"
        ),
        factory=lambda exc, provider: EmbedInputTooLargeError(
            str(exc), provider=provider
        ),
    ),
]


def map_error(exc: Exception, provider: str | None = None) -> EmbedError:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return EmbedAuthenticationError(str(exc), provider=provider)
    if status_code == 429:
        return EmbedRateLimitError(str(exc), provider=provider)

    for rule in _GLOBAL_RULES:
        if rule.classifier(exc):
            return rule.factory(exc, provider)

    return EmbedAPIError(str(exc), status_code=status_code, provider=provider)


def is_retriable(error: BaseException) -> bool:
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit)):
        return False

    if isinstance(error, (asyncio.TimeoutError, aiohttp.ClientError)):
        return True
    if isinstance(error, EmbedTimeoutError):
        return True
    if isinstance(error, EmbedRateLimitError):
        return True
    if isinstance(error, EmbedAuthenticationError):
        return False
    if isinstance(error, EmbedConfigError):
        return False
    if isinstance(error, EmbedInputTooLargeError):
        return False

    if isinstance(error, EmbedAPIError):
        status_code = error.status_code
        if status_code:
            if status_code >= 500 or status_code == 429:
                return True
            if 400 <= status_code < 500:
                return False

        return True

    return True
