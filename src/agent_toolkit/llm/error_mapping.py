"""Provider errors in, taxonomy out -- and the retry decision that follows.

``map_error`` is copied from ``agent-evaluation``'s ``error_mapping.py``;
``is_retriable`` is its ``factory.py:63`` ``_is_retriable_error``, moved here
because the two halves of error classification belong in one place and T6's
criteria test both. Neither changes behavior: the spec pins retry classification
as carrying over unchanged.

Two departures from the original file, both mechanical:

- ``openai`` is imported unconditionally. The whole ``llm`` subpackage requires
  the extra, so the original's "SDK not installed" fallback -- a module-level
  ``openai = None`` and a rule list spliced together afterwards -- is
  unreachable here, and dropping it removes the ``type: ignore`` it needed.
- ``aiohttp`` moves from a function-local import to a module-level one, for the
  same reason.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp
import openai

from agent_toolkit.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    ProviderContextWindowError,
)

__all__ = ["is_retriable", "map_error"]


ErrorClassifier = Callable[[Exception], bool]


@dataclass(frozen=True)
class MappingRule:
    classifier: ErrorClassifier
    factory: Callable[[Exception, str | None], LLMError]


def _instance_of(*types: type[BaseException]) -> ErrorClassifier:
    return lambda exc: isinstance(exc, types)


def _message_contains(*needles: str) -> ErrorClassifier:
    def _classifier(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(needle in msg for needle in needles)

    return _classifier


# Order is the original's: the SDK-type rules were spliced in front of the
# message rules, so a typed match wins over a substring match.
_GLOBAL_RULES: list[MappingRule] = [
    MappingRule(
        classifier=_instance_of(openai.AuthenticationError),
        factory=lambda exc, provider: LLMAuthenticationError(
            str(exc), provider=provider
        ),
    ),
    MappingRule(
        classifier=_instance_of(openai.RateLimitError),
        factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
    ),
    MappingRule(
        classifier=_message_contains("rate limit", "429", "quota"),
        factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
    ),
    MappingRule(
        classifier=_message_contains("context length", "maximum context"),
        factory=lambda exc, provider: ProviderContextWindowError(
            str(exc), provider=provider
        ),
    ),
]


def map_error(exc: Exception, provider: str | None = None) -> LLMError:
    """Map provider-specific errors to unified internal exceptions."""
    # Heuristic check for status codes before rules
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return LLMAuthenticationError(str(exc), provider=provider)
    if status_code == 429:
        return LLMRateLimitError(str(exc), provider=provider)

    for rule in _GLOBAL_RULES:
        if rule.classifier(exc):
            return rule.factory(exc, provider)

    return LLMAPIError(str(exc), status_code=status_code, provider=provider)


def is_retriable(error: BaseException) -> bool:
    """Whether retrying ``error`` could plausibly succeed.

    Retriable: timeouts, rate limits (429), server errors (5xx), and connection
    errors. Not retriable: cancellation, authentication, configuration, and 4xx
    other than 429.

    An unrecognized exception is retriable. That is the original's choice and it
    is the right way round for this client: the openai SDK raises
    ``httpx``-based connection errors that no branch below names, and they are
    exactly the transient failures retrying exists for.
    """
    if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit)):
        return False

    if isinstance(error, (asyncio.TimeoutError, aiohttp.ClientError)):
        return True
    if isinstance(error, LLMTimeoutError):
        return True
    if isinstance(error, LLMRateLimitError):
        return True
    if isinstance(error, LLMAuthenticationError):
        return False  # Don't retry auth errors
    if isinstance(error, LLMConfigError):
        return False

    if isinstance(error, LLMAPIError):
        status_code = error.status_code
        if status_code:
            # Retry on server errors (5xx) and rate limits (429)
            if status_code >= 500 or status_code == 429:
                return True
            # Don't retry on client errors (4xx except 429)
            if 400 <= status_code < 500:
                return False

        # When status_code is None (e.g. connection drop), retry.
        return True

    # For other exceptions (network errors, etc.), retry
    return True
