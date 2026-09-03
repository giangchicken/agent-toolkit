"""logic. What a model name says about family, capability and cost."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import tiktoken

__all__ = [
    "FAMILY_MARKERS",
    "NATIVE_TOOL_CALLING_PATTERNS",
    "REASONING_PATTERNS",
    "UNKNOWN_FAMILY",
    "count_tokens",
    "model_family",
    "supports_native_tool_calling",
    "supports_reasoning",
]

UNKNOWN_FAMILY = "unknown"

FAMILY_MARKERS: list[tuple[str, str]] = [
    ("gemma", "gemma"),
    ("glm", "glm"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("gpt", "gpt"),
    ("gemini", "gemini"),
    ("llama", "llama"),
    ("llama", "vicuna"),
]

REASONING_PATTERNS = [
    r"^qwen3-.*",
    r"^qwen3\..*",
    r"^glm-4.*",
    r"^glm-5.*",
    r"^gpt-oss-.*",
    r"^gpt-5.*",
    r"^deepseek-.*",
]

NATIVE_TOOL_CALLING_PATTERNS = [
    r"^glm-4.*",
    r"^glm-5.*",
    r"^gpt-4.*",
    r"^gpt-5.*",
]

_CHAT_FORMAT_OVERHEAD = 4


def model_family(name: str) -> str:
    lowered = name.lower()
    for family, marker in FAMILY_MARKERS:
        if marker in lowered:
            return family
    return UNKNOWN_FAMILY


def supports_reasoning(name: str) -> bool:
    lowered = name.lower()
    return any(re.match(pattern, lowered) for pattern in REASONING_PATTERNS)


def supports_native_tool_calling(name: str) -> bool:
    lowered = name.lower()
    return any(re.match(pattern, lowered) for pattern in NATIVE_TOOL_CALLING_PATTERNS)


def count_tokens(
    messages: Sequence[Mapping[str, Any]], model: str | None = None
) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model) if model else None
    except KeyError:
        encoding = None
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for message in messages:
        total += _CHAT_FORMAT_OVERHEAD
        for value in message.values():
            if value:
                total += len(encoding.encode(str(value)))
    return total
