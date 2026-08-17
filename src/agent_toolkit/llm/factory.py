"""One LLM call: resolve settings, throttle, attempt, retry, map the error.

This is the harvested ``factory.py``'s ``complete()`` with its two host couplings
cut. The retry defaults no longer come from ``src.dependencies.settings`` (see
:mod:`agent_toolkit.llm.retry`), and the traffic controller no longer arrives
inside a process-cached ``LLMConfig`` but is fetched per event loop from
:func:`agent_toolkit.llm.get_traffic_controller`.

Three changes beyond that, each visible in ``tests/test_llm_complete.py``:

- ``reraise=True``. Without it tenacity raises its own ``RetryError`` once the
  attempts run out, so the error a caller sees on exhaustion is *not* from the
  ``LLMError`` taxonomy the rest of this package builds -- the mapped exception
  is buried in ``RetryError.last_attempt``. Every other outcome raises an
  ``LLMError``; exhaustion did not.
- The retry predicate is ``is_retriable`` alone. The original OR-ed it with
  ``retry_if_exception_type(LLMRateLimitError) | retry_if_exception_type(LLMTimeoutError)``,
  both of which ``is_retriable`` already answers True for.
- ``max_retries`` / ``retry_delay`` / ``exponential_backoff`` are rejected rather
  than accepted. They are ``RetryPolicy`` fields now, and ``**kwargs`` here goes
  into the request body -- so silently accepting them would ship ``max_retries``
  to the provider as a chat-completion parameter and quietly use the default
  policy. Five call sites in ``agent-evaluation`` pass ``max_retries=``; a
  ``TypeError`` naming the replacement is what they should meet.

``complete_with_reasoning`` is the function; ``complete`` is it with the reasoning
dropped. The harvested code had only the second, and suppressed thinking on any
``qwen3*`` model to keep it from polluting the answer -- so reasoning was
unavailable by construction. It is now a parameter (``enable_thinking``, unset by
default) and a return field.

``LLMConfig.max_tokens``, ``temperature``, ``max_concurrency`` and
``requests_per_minute`` are read here. In the harvested code they were read by
nobody: ``_resolve_config`` did not return them and every call site passed its
own. A resolver that sets them now has them honored, and a caller's explicit
argument still wins.
"""

from typing import Any

import tenacity

from agent_toolkit.llm.config import resolve_config
from agent_toolkit.llm.error_mapping import is_retriable, map_error
from agent_toolkit.llm.exceptions import LLMConfigError
from agent_toolkit.llm.executors import Completion, sdk_complete
from agent_toolkit.llm.retry import RetryPolicy, backoff, get_default_retry_policy
from agent_toolkit.llm.traffic_control import get_traffic_controller
from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = ["complete", "complete_with_reasoning"]

_REPLACED_BY_POLICY = ("max_retries", "retry_delay", "exponential_backoff")


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
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> Completion:
    """Complete ``prompt`` against an OpenAI-compatible endpoint.

    Args:
        prompt: The user prompt. Ignored if ``messages`` is given.
        system_prompt: Prepended as a system message. Ignored if ``messages`` is given.
        model: Model name, and the key the installed resolver looks up.
        api_key: Overrides the resolved key. ``""`` is respected as "no key".
        base_url: Overrides the resolved endpoint.
        api_version: Accepted and unused. It configures the Azure client, and the
            Azure binding is out of scope for v0.1; the harvested code accepted
            and ignored it identically. Named explicitly rather than left to
            ``**kwargs`` so it cannot reach the request body.
        binding: Provider label, used for the ``provider`` on mapped errors.
        messages: A pre-built message list, used instead of ``prompt``.
        extra_headers: Merged over the resolved headers for this call only.
        reasoning_effort: Passed through when set.
        enable_thinking: Sent as the ``enable_thinking`` chat-template kwarg when
            set; ``True`` asks for reasoning, ``False`` suppresses it. Left
            ``None`` -- the default -- nothing is sent and the server decides.
            **Do not assume that default.** A self-hosted ``gemma-4-31B-it``
            returned no reasoning at all until asked explicitly, and then returned
            459 characters of it on the same prompt; Qwen3's template defaults the
            other way. Sent for any model, not just the Qwen line the harvested
            code special-cased: that endpoint accepted the kwarg for a gemma
            model, and a server that does not know it answers 400, which is a
            clearer outcome than silently dropping what the caller asked for.
        retry: Overrides the process-wide :class:`RetryPolicy` for this call.
        **kwargs: Extra chat-completion parameters (``temperature``,
            ``max_tokens``, ``top_p``, …). Forwarded to the provider verbatim.

    Returns:
        A :class:`Completion`. ``content`` is the answer with any inline
        ``<think>`` block removed, ``""`` if the response carries no extractable
        text; ``reasoning`` is the reasoning field or that inline block, ``""``
        when there was none.

    Raises:
        LLMConfigError: No model, or the resolver could not produce a config.
        LLMError: Any provider failure, mapped to the taxonomy, after the retry
            policy is exhausted.
    """
    for name in _REPLACED_BY_POLICY:
        if name in kwargs:
            raise TypeError(
                f"complete() no longer takes {name!r}: pass "
                f"retry=RetryPolicy({name}=...) or call set_default_retry_policy() once"
            )

    config = resolve_config(
        model=model,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        binding=binding,
        extra_headers=extra_headers,
        reasoning_effort=reasoning_effort,
    )
    policy = retry if retry is not None else get_default_retry_policy()
    controller = get_traffic_controller(
        config.model,
        max_concurrency=config.max_concurrency,
        requests_per_minute=config.requests_per_minute,
    )

    kwargs.setdefault("max_tokens", config.max_tokens)
    kwargs.setdefault("temperature", config.temperature)

    def log_retry(retry_state: tenacity.RetryCallState) -> None:
        outcome = retry_state.outcome
        error = outcome.exception() if outcome is not None else None
        message = str(error) if error is not None else "unknown error"
        if not message.strip():
            message = f"{type(error).__name__} (no message)"
        logger.warning(
            "LLM call failed (attempt %s/%s), retrying in %.1fs: %s",
            retry_state.attempt_number,
            policy.max_retries + 1,
            retry_state.upcoming_sleep,
            message,
        )

    @tenacity.retry(
        retry=tenacity.retry_if_exception(is_retriable),
        wait=backoff(policy),
        stop=tenacity.stop_after_attempt(policy.max_retries + 1),
        before_sleep=log_retry,
        reraise=True,
    )
    async def attempt() -> Completion:
        # The controller is entered per attempt, so a failed attempt returns its
        # concurrency slot before the retry sleeps rather than holding it for the
        # whole backoff.
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

    return await attempt()


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
    retry: RetryPolicy | None = None,
    **kwargs: Any,
) -> str:
    """The answer text from :func:`complete_with_reasoning`, and nothing else.

    The signature is spelled out rather than forwarded through ``*args`` because
    the spec pins it for the ``agent-evaluation`` migration: ``str`` in, ``str``
    out, every parameter named and type-checked at the call site. Reasoning is
    reachable only through the sibling above, so a caller that wants it changes
    one name and a caller that does not is unaffected.
    """
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
        retry=retry,
        **kwargs,
    )
    return completion.content
