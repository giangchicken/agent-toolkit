"""The one provider call: a non-streaming OpenAI-compatible chat completion.

Copied from ``agent-evaluation``'s ``executors.py`` with the host logger import
replaced (coupling #3) and three removals:

- ``provider_name``, a required keyword argument the function never read.
- ``api_version``, accepted and ignored -- see :func:`agent_toolkit.llm.complete`,
  which still accepts it because every current call site passes it.
- ``sdk_stream`` and ``sdk_complete_with_tools``, deferred to v0.2 along with the
  ``stream`` and ``complete_with_tools`` that call them.

One addition: a caller's ``extra_body`` is merged rather than assigned, so the
Qwen thinking switch below survives a call that also passes ``guided_json``.

``extract_response_content`` comes from the same component's ``utils.py`` and
lands here rather than in a module of its own: it exists to read one shape, the
message object this function has just received.

A fresh ``AsyncOpenAI`` per call is the harvested behavior, kept: the
``x-session-affinity`` header is per-call and ``default_headers`` is per-client,
so caching the client would mean moving that header to the request. The cost is
that connections are never reused across calls.
"""

import re
import uuid
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

__all__ = ["extract_response_content", "sdk_complete"]


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


_QWEN_THINKING_RE = re.compile(r"^qwen3[-.].*", re.IGNORECASE)


def _is_thinking_model(model: str) -> bool:
    """Return True if *model* honors the ``enable_thinking`` chat-template kwarg.

    Gates the qwen-specific ``extra_body`` below, so scoped to the Qwen3
    family only — NOT every reasoning-capable model. GLM/gpt-5/deepseek are
    reasoning-capable but stream their reasoning via the ``reasoning`` field
    and must not receive this kwarg (GLM endpoint behavior on this kwarg is
    unverified).
    """
    return bool(_QWEN_THINKING_RE.match(model))


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
    **kwargs: Any,
) -> str:
    """Non-streaming completion using the openai SDK.

    ``max_retries=0`` on the client is load-bearing: retry is owned by
    :func:`agent_toolkit.llm.complete`, which classifies the error first.
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
    if _is_thinking_model(model):
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    # A caller's ``extra_body`` merges into the one above instead of replacing
    # it. ``complete_structured(mode="grammar")`` puts ``guided_json`` there, and
    # a plain ``update`` would drop ``enable_thinking`` on a Qwen3 model -- the
    # answer would come back in ``reasoning``, where nothing reads it.
    caller_extra_body = kwargs.pop("extra_body", None)
    payload.update(kwargs)
    if caller_extra_body is not None:
        merged: dict[str, Any] = dict(payload.get("extra_body") or {})
        merged.update(caller_extra_body)
        payload["extra_body"] = merged

    response = await client.chat.completions.create(**payload)
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message")
    return extract_response_content(message)
