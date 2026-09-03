"""adapter. The OpenAI SDK's chat response to a Completion."""

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

_REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning: str = ""


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

    if isinstance(message, (int, float, bool)):
        return str(message)
    return ""


def extract_reasoning_content(message: object) -> str:
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
    timeout: float = 120.0,
    **kwargs: Any,
) -> Completion:
    default_headers: dict[str, str] = {"x-session-affinity": uuid.uuid4().hex}
    if extra_headers:
        default_headers.update(extra_headers)

    client = AsyncOpenAI(
        api_key=api_key or "no-key",
        base_url=base_url,
        default_headers=default_headers,
        max_retries=0,
        timeout=timeout,
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

    if enable_thinking is not None:
        payload["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

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

    inline_reasoning, content = split_thinking(extract_response_content(message))
    field_reasoning = extract_reasoning_content(message)
    return Completion(content=content, reasoning=field_reasoning or inline_reasoning)
