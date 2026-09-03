"""wiring. One chat call: resolve, throttle, attempt, retry, map."""

import time
from typing import Any

import tenacity

from agent_toolkit.llm.config import resolve_config
from agent_toolkit.llm.error_mapping import is_retriable, map_error
from agent_toolkit.llm.exceptions import LLMConfigError
from agent_toolkit.llm.executors import Completion, sdk_complete
from agent_toolkit.llm.traffic_control import get_traffic_controller
from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_EXPONENTIAL_BACKOFF",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "MAX_BACKOFF",
    "complete",
    "complete_with_reasoning",
]

DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_EXPONENTIAL_BACKOFF = True

MAX_BACKOFF = 120.0


async def complete_with_reasoning(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str | None = None,
    messages: list[dict[str, object]] | None = None,
    extra_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    **kwargs: Any,
) -> Completion:
    config = resolve_config(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        binding=binding,
        extra_headers=extra_headers,
        reasoning_effort=reasoning_effort,
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

    kwargs.setdefault("max_tokens", config.max_tokens)
    kwargs.setdefault("temperature", config.temperature)
    kwargs.setdefault("timeout", config.timeout)
    if config.top_p is not None:
        kwargs.setdefault("top_p", config.top_p)
    if enable_thinking is None:
        enable_thinking = config.enable_thinking

    def log_retry(retry_state: tenacity.RetryCallState) -> None:
        outcome = retry_state.outcome
        error = outcome.exception() if outcome is not None else None
        message = str(error) if error is not None else "unknown error"
        if not message.strip():
            message = f"{type(error).__name__} (no message)"
        logger.warning(
            "LLM call failed (attempt %s/%s), retrying in %.1fs: %s",
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
    async def attempt() -> Completion:
        try:
            async with controller:
                return await sdk_complete(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=config.model,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    messages=messages,
                    extra_headers=config.extra_headers or None,
                    reasoning_effort=config.reasoning_effort,
                    enable_thinking=enable_thinking,
                    **kwargs,
                )
        except LLMConfigError:
            raise
        except Exception as exc:
            raise map_error(exc, provider=config.binding) from exc

    logger.debug(
        "LLM call: model=%s binding=%s base_url=%s max_tokens=%s temperature=%s "
        "reasoning_effort=%s enable_thinking=%s max_retries=%s",
        config.model,
        config.binding,
        config.base_url,
        kwargs.get("max_tokens"),
        kwargs.get("temperature"),
        config.reasoning_effort,
        enable_thinking,
        max_retries,
    )
    started = time.perf_counter()
    try:
        completion = await attempt()
    except Exception as exc:
        logger.error(
            "LLM call failed after %.2fs: model=%s %s: %s",
            time.perf_counter() - started,
            config.model,
            type(exc).__name__,
            exc,
        )
        raise
    logger.debug(
        "LLM call ok in %.2fs: model=%s content=%d chars reasoning=%d chars",
        time.perf_counter() - started,
        config.model,
        len(completion.content),
        len(completion.reasoning),
    )
    return completion


async def complete(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str | None = None,
    messages: list[dict[str, object]] | None = None,
    extra_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    **kwargs: Any,
) -> str:
    completion = await complete_with_reasoning(
        prompt,
        system_prompt=system_prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        binding=binding,
        messages=messages,
        extra_headers=extra_headers,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
        max_retries=max_retries,
        retry_delay=retry_delay,
        exponential_backoff=exponential_backoff,
        **kwargs,
    )
    return completion.content
