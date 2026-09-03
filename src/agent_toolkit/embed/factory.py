"""wiring. One embeddings call: resolve, batch, throttle, retry, map."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import tenacity

from agent_toolkit.embed.config import resolve_config
from agent_toolkit.embed.error_mapping import is_retriable, map_error
from agent_toolkit.embed.exceptions import EmbedConfigError, EmbedProviderError
from agent_toolkit.embed.executors import sdk_embed
from agent_toolkit.embed.traffic_control import get_traffic_controller
from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_EXPONENTIAL_BACKOFF",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "MAX_BACKOFF",
    "vectors",
]

DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_EXPONENTIAL_BACKOFF = True

MAX_BACKOFF = 120.0


def split_into_batches(
    texts: Sequence[str], batch_size: int | None
) -> list[Sequence[str]]:
    if batch_size is None:
        return [texts]
    if batch_size < 1:
        raise ValueError(
            f"batch_size is {batch_size!r}; a batch carries at least one text"
        )
    return [
        texts[start : start + batch_size] for start in range(0, len(texts), batch_size)
    ]


async def vectors(
    texts: Sequence[str],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    binding: str | None = None,
    extra_headers: dict[str, str] | None = None,
    batch_size: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    **kwargs: Any,
) -> list[list[float]]:
    if not texts:
        return []

    config = resolve_config(
        model=model,
        api_key=api_key,
        base_url=base_url,
        binding=binding,
        extra_headers=extra_headers,
    )
    wait_strategy = (
        tenacity.wait_exponential(
            multiplier=retry_delay,
            min=retry_delay,
            max=MAX_BACKOFF,
        )
        if exponential_backoff
        else tenacity.wait_fixed(retry_delay)
    )
    controller = get_traffic_controller(
        config.model,
        max_concurrency=config.max_concurrency,
        requests_per_minute=config.requests_per_minute,
    )

    kwargs.setdefault("timeout", config.timeout)

    def log_retry(retry_state: tenacity.RetryCallState) -> None:
        outcome = retry_state.outcome
        error = outcome.exception() if outcome is not None else None
        message = str(error) if error is not None else "unknown error"
        if not message.strip():
            message = f"{type(error).__name__} (no message)"
        logger.warning(
            "embeddings call failed (attempt %s/%s), retrying in %.1fs: %s",
            retry_state.attempt_number,
            max_retries + 1,
            retry_state.upcoming_sleep,
            message,
        )

    @tenacity.retry(
        retry=tenacity.retry_if_exception(is_retriable),
        wait=wait_strategy,
        stop=tenacity.stop_after_attempt(max_retries + 1),
        before_sleep=log_retry,
        reraise=True,
    )
    async def attempt(batch: Sequence[str]) -> list[list[float]]:
        try:
            async with controller:
                return await sdk_embed(
                    texts=batch,
                    model=config.model,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    extra_headers=config.extra_headers or None,
                    **kwargs,
                )
        except EmbedConfigError:
            raise
        except Exception as exc:
            raise map_error(exc, provider=config.binding) from exc

    batches = split_into_batches(texts, batch_size)
    logger.debug(
        "embeddings call: model=%s base_url=%s texts=%d batches=%d max_retries=%s",
        config.model,
        config.base_url,
        len(texts),
        len(batches),
        max_retries,
    )
    started = time.perf_counter()
    try:
        answered = await asyncio.gather(*(attempt(batch) for batch in batches))
    except Exception as exc:
        logger.error(
            "embeddings call failed after %.2fs: model=%s texts=%d %s: %s",
            time.perf_counter() - started,
            config.model,
            len(texts),
            type(exc).__name__,
            exc,
        )
        raise
    vectors = [vector for batch in answered for vector in batch]
    logger.debug(
        "embeddings call ok in %.2fs: model=%s vectors=%d dimensions=%s",
        time.perf_counter() - started,
        config.model,
        len(vectors),
        len(vectors[0]) if vectors else 0,
    )

    if len(vectors) != len(texts):
        raise EmbedProviderError(
            f"{config.model!r} was asked to embed {len(texts)} texts and answered "
            f"with {len(vectors)} vectors; the pairing is positional, so a short "
            "answer is a shifted list rather than a missing one",
            provider=config.binding,
        )
    return vectors
