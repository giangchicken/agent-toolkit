"""logic. Streaming a top-level JSON array without loading it whole."""

import json
from collections.abc import Iterator
from typing import IO, Any

from agent_toolkit.errors import ToolkitError

__all__ = ["DEFAULT_BUFFER_SIZE", "iter_json_array"]

DEFAULT_BUFFER_SIZE = 1 << 20


def _consume_opening_bracket(fp: IO[str], buffer_size: int) -> str:
    buf = ""
    while True:
        buf = buf.lstrip()
        if buf:
            if buf[0] != "[":
                raise ToolkitError(
                    f"top-level JSON value is not an array: it starts with {buf[0]!r}"
                )
            return buf[1:]
        chunk = fp.read(buffer_size)
        if not chunk:
            raise ToolkitError("expected a JSON array, found no content")
        buf = chunk


def iter_json_array(
    fp: IO[str], *, buffer_size: int = DEFAULT_BUFFER_SIZE
) -> Iterator[Any]:
    if buffer_size <= 0:
        raise ToolkitError(f"buffer_size must be positive, got {buffer_size}")

    decoder = json.JSONDecoder()
    buf = _consume_opening_bracket(fp, buffer_size)
    expect_separator = False

    while True:
        buf = buf.lstrip()

        if not buf:
            chunk = fp.read(buffer_size)
            if not chunk:
                raise ToolkitError(
                    "unterminated JSON array: input ended before the closing ']'"
                )
            buf = chunk
            continue

        if buf[0] == "]":
            return

        if expect_separator:
            if buf[0] != ",":
                raise ToolkitError(
                    f"expected ',' or ']' after an element, found {buf[0]!r}"
                )
            buf = buf[1:]
            expect_separator = False
            continue

        try:
            element, end = decoder.raw_decode(buf)
        except ValueError as exc:
            chunk = fp.read(buffer_size)
            if not chunk:
                raise ToolkitError(f"malformed or truncated JSON array: {exc}") from exc
            buf += chunk
            continue

        yield element
        buf = buf[end:]
        expect_separator = True
