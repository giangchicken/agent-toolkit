"""adapter. The OpenAI SDK's embeddings response to a list of vectors."""

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

__all__ = ["sdk_embed"]


async def sdk_embed(
    *,
    texts: Sequence[str],
    model: str,
    api_key: str | None,
    base_url: str | None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> list[list[float]]:
    client = AsyncOpenAI(
        api_key=api_key or "no-key",
        base_url=base_url,
        default_headers=dict(extra_headers) if extra_headers else None,
        max_retries=0,
        timeout=timeout,
    )

    payload: dict[str, Any] = {"model": model, "input": list(texts)}
    payload.update(kwargs)

    answered = await client.embeddings.create(**payload)
    return [row.embedding for row in sorted(answered.data, key=lambda row: row.index)]
