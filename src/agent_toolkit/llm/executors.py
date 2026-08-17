"""The one provider call: a non-streaming OpenAI-compatible chat completion.

Copied from ``agent-evaluation``'s ``executors.py`` with the host logger import
replaced (coupling #3) and three removals:

- ``provider_name``, a required keyword argument the function never read.
- ``api_version``, accepted and ignored -- see :func:`agent_toolkit.llm.complete`,
  which still accepts it because every current call site passes it.
- ``sdk_stream`` and ``sdk_complete_with_tools``, deferred to v0.2 along with the
  ``stream`` and ``complete_with_tools`` that call them.

Two additions:

- A caller's ``extra_body`` is merged rather than assigned, so an explicit
  thinking switch survives a call that also passes ``guided_json``.
- The harvested code force-fed ``enable_thinking: False`` to any ``qwen3*`` model
  and returned a bare ``str``, so reasoning was suppressed where it could be and
  dropped where it could not. Thinking is now the caller's decision
  (``enable_thinking``, unset by default) and the reasoning comes back on
  :class:`Completion` next to the answer rather than in front of it.

``extract_response_content`` comes from the same component's ``utils.py`` and
lands here rather than in a module of its own: it exists to read one shape, the
message object this function has just received.

A fresh ``AsyncOpenAI`` per call is the harvested behavior, kept: the
``x-session-affinity`` header is per-call and ``default_headers`` is per-client,
so caching the client would mean moving that header to the request. The cost is
that connections are never reused across calls.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from agent_toolkit.string_utils import split_thinking

__all__ = [
    "Completion",
    "extract_reasoning_content",
    "extract_response_content",
    "sdk_complete",
]


@dataclass(frozen=True)
class Completion:
    """One response, with reasoning separated from the answer.

    ``reasoning`` is ``""`` when the model emitted none, and carries its markers
    when it came from a ``<think>``-style block. ``content`` never contains
    reasoning: it is the answer with any inline block removed.
    """

    content: str
    reasoning: str = ""


# Two spellings in the wild. vLLM's ``--reasoning-parser`` and DeepSeek's API use
# ``reasoning_content``; OpenRouter and several gateways use ``reasoning``. Both
# arrive as extra fields on the SDK's message model, which allows them.
#
# ``reasoning_content`` is first because it is the one the endpoint this package
# is built for returns: a self-hosted ``gemma-4-31B-it`` answered with
# ``{"content", "reasoning_content", "role"}`` and no ``reasoning`` key.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _build_messages(
    *,
    prompt: str,
    system_prompt: str | None,
    messages: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if messages:
        return messages
    msgs: list[dict[str, object]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _extract_content_field(content: object) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def extract_response_content(message: object) -> str:
    """Extract textual content from response payloads.

    Returns empty string when the message carries no meaningful text
    (e.g. a streaming delta with ``content=None``).  Never falls back
    to ``str(message)`` for complex objects — that would inject garbage
    like ``"{'provider_specific_fields': None, ...}"`` into the response
    stream and corrupt downstream JSON parsing.
    """
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    if isinstance(message, Mapping):
        content = _extract_content_field(message.get("content"))
        if content:
            return content
        if "text" in message and message["text"] is not None:
            return str(message["text"])
        return ""

    # OpenAI SDK response models expose attributes instead of dict keys.
    if hasattr(message, "content"):
        content = _extract_content_field(message.content)
        if content:
            return content
    if hasattr(message, "text"):
        text_value = message.text
        if text_value is not None:
            return str(text_value)

    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
        except Exception:
            dumped = None
        if dumped is not None and dumped is not message:
            return extract_response_content(dumped)

    # Only stringify simple/primitive values; complex SDK objects with no
    # extractable content should yield empty string, not their repr.
    if isinstance(message, (int, float, bool)):
        return str(message)
    return ""


def extract_reasoning_content(message: object) -> str:
    """Read a separate reasoning field off a response message, or ``""``.

    Checks ``reasoning_content`` then ``reasoning``. Returns ``""`` for a message
    that carries neither, which is every non-reasoning model and every reasoning
    model served without a reasoning parser -- those put it inline in ``content``
    instead, where :func:`agent_toolkit.string_utils.split_thinking` finds it.
    """
    if message is None:
        return ""

    for field in _REASONING_FIELDS:
        value = (
            message.get(field)
            if isinstance(message, Mapping)
            else getattr(message, field, None)
        )
        if value:
            return _extract_content_field(value)
    return ""


async def sdk_complete(
    *,
    prompt: str,
    system_prompt: str | None = None,
    model: str,
    api_key: str | None,
    base_url: str | None,
    messages: list[dict[str, object]] | None = None,
    extra_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    **kwargs: Any,
) -> Completion:
    """Non-streaming completion using the openai SDK.

    ``max_retries=0`` on the client is load-bearing: retry is owned by
    :func:`agent_toolkit.llm.complete`, which classifies the error first.

    ``enable_thinking`` is sent as a chat-template kwarg only when it is not
    ``None``. Left alone, the server's own default decides, and that default is
    not predictable from the model name: a self-hosted ``gemma-4-31B-it`` returned
    no reasoning until asked, while Qwen3 templates default it on.
    """

    default_headers: dict[str, str] = {"x-session-affinity": uuid.uuid4().hex}
    if extra_headers:
        default_headers.update(extra_headers)

    client = AsyncOpenAI(
        api_key=api_key or "no-key",
        base_url=base_url,
        default_headers=default_headers,
        max_retries=0,
        timeout=120.0,
    )

    max_tokens_val = int(kwargs.pop("max_tokens", 4096))
    temperature_val = float(kwargs.pop("temperature", 0.7))

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
        ),
        "temperature": temperature_val,
        "max_tokens": max_tokens_val,
    }

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Disable thinking mode for reasoning models (e.g. Qwen3.6) so that
    # the response is in `content` instead of `reasoning`.
    if enable_thinking is not None:
        payload["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    # A caller's ``extra_body`` merges into the one above instead of replacing
    # it. ``complete_structured(mode="grammar")`` puts ``guided_json`` there, and
    # a plain ``update`` would drop an explicit ``enable_thinking``.
    caller_extra_body = kwargs.pop("extra_body", None)
    payload.update(kwargs)
    if caller_extra_body is not None:
        merged: dict[str, Any] = dict(payload.get("extra_body") or {})
        merged.update(caller_extra_body)
        payload["extra_body"] = merged

    response = await client.chat.completions.create(**payload)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return Completion(content="")
    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message")

    # Both shapes, because a model can use either and some servers use both: the
    # separate field if there is one, and whatever an inline block holds. The
    # inline split runs either way, so ``content`` is the answer alone.
    inline_reasoning, content = split_thinking(extract_response_content(message))
    field_reasoning = extract_reasoning_content(message)
    return Completion(content=content, reasoning=field_reasoning or inline_reasoning)
