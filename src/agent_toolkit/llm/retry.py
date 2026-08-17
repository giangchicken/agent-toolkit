"""How many attempts, and how long between them.

Host coupling #2 lived here. The harvested ``factory.py:35`` read its retry
defaults out of the application that installed it::

    from src.dependencies import settings
    DEFAULT_MAX_RETRIES = settings.retry.max_retries

A library cannot do that. The three numbers are the same ones -- ``8`` retries,
``5.0`` second base delay, exponential -- stated here as defaults instead of
fetched from a host object, and overridable per call or once per process.

``is_retriable`` is the other half of retry, and the spec's layout puts it in
this module. It is in :mod:`agent_toolkit.llm.error_mapping` instead, next to
``map_error``: "what kind of error is this" and "is that kind worth retrying"
read one taxonomy, and T6 tested them together.
"""

from collections.abc import Callable
from dataclasses import dataclass

import tenacity
from tenacity.wait import wait_base

from agent_toolkit.llm.exceptions import LLMRateLimitError

__all__ = [
    "RetryPolicy",
    "backoff",
    "get_default_retry_policy",
    "set_default_retry_policy",
]

# Cap on any single wait. The harvested code's `max=120`, unchanged, and not a
# policy field: the plan names three knobs and three is what callers get.
_MAX_BACKOFF = 120.0


@dataclass(frozen=True)
class RetryPolicy:
    """Defaults match ``LLMRetryConfig`` in the host this was harvested from.

    ``base_delay`` is both the first wait and the multiplier: attempt *n* waits
    ``base_delay * 2 ** (n - 1)``, capped at two minutes. With
    ``exponential_backoff=False`` every wait is ``base_delay``.
    """

    max_retries: int = 8
    base_delay: float = 5.0
    exponential_backoff: bool = True


_default = RetryPolicy()


def set_default_retry_policy(policy: RetryPolicy | None) -> None:
    """Install the process-wide policy. ``None`` restores the defaults."""
    global _default
    _default = policy if policy is not None else RetryPolicy()


def get_default_retry_policy() -> RetryPolicy:
    """The policy :func:`agent_toolkit.llm.complete` uses when passed none."""
    return _default


def backoff(policy: RetryPolicy) -> Callable[[tenacity.RetryCallState], float]:
    """Build the wait function for ``policy``, floored by ``retry_after``.

    The floor is the spec's contract -- "``LLMRateLimitError.retry_after`` raises
    the floor on the computed backoff" -- and the harvested ``complete()`` did
    not have it; only ``stream()`` did, at ``factory.py:428``. Note that nothing
    in this library populates ``retry_after`` yet: ``map_error`` builds
    ``LLMRateLimitError`` without reading the response's ``Retry-After`` header,
    so today the floor applies only to a rate-limit error a caller raised itself.
    ``tests/test_llm_complete.py`` pins that gap so closing it is a visible
    change rather than a silent one.
    """
    base: wait_base = (
        tenacity.wait_exponential(
            multiplier=policy.base_delay,
            min=policy.base_delay,
            max=_MAX_BACKOFF,
        )
        if policy.exponential_backoff
        else tenacity.wait_fixed(policy.base_delay)
    )

    def wait(retry_state: tenacity.RetryCallState) -> float:
        delay = base(retry_state)
        outcome = retry_state.outcome
        error = outcome.exception() if outcome is not None else None
        if isinstance(error, LLMRateLimitError) and error.retry_after:
            return max(delay, error.retry_after)
        return delay

    return wait
