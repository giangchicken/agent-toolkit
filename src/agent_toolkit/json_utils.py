"""Streaming iteration over a large top-level JSON array.

``json.load`` on the pipeline's 126 MB training corpus costs roughly 1.5 GB
resident, because every one of its 21,172 objects is materialized at once. This
module reads a bounded buffer instead and applies ``json.JSONDecoder.raw_decode``
at each value boundary, so peak memory tracks the buffer plus the single largest
element rather than the file.

Pure stdlib is deliberate: ``ijson`` would put a C extension in the dependency
path of every consumer for one function.

Unlike the rest of the core, this module raises. A malformed 126 MB import must
fail loudly, because the alternative -- yielding a truncated dataset that looks
complete -- produces a silently undersized training set, and nothing downstream
would notice.
"""

import json
from collections.abc import Iterator
from typing import IO, Any

from agent_toolkit.errors import ToolkitError

__all__ = ["DEFAULT_BUFFER_SIZE", "iter_json_array"]

DEFAULT_BUFFER_SIZE = 1 << 20


def _consume_opening_bracket(fp: IO[str], buffer_size: int) -> str:
    """Read past leading whitespace and the opening ``[``; return what follows."""
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
        # buf is empty after the lstrip above, so assignment is the append.
        buf = chunk


def iter_json_array(
    fp: IO[str], *, buffer_size: int = DEFAULT_BUFFER_SIZE
) -> Iterator[Any]:
    """Yield elements of a top-level JSON array one at a time.

    Memory is bounded by ``buffer_size`` plus the largest single element.

    Being a generator, nothing is read and no error is raised until iteration
    starts. Raises :class:`ToolkitError` if the top-level value is not an array,
    if the array is unterminated, or if an element does not parse once the whole
    input has been read.

    A trailing comma before the closing bracket is tolerated rather than
    rejected: it is invalid JSON, but it cannot cause a short iteration, which is
    the failure this function exists to prevent.
    """
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
            # Consume the comma and go back to the top, so the whitespace *after*
            # it is stripped by the same lstrip that handles a buffer refill.
            # Stripping only before the comma leaves raw_decode looking at
            # whitespace, which it does not skip: it raises "Expecting value" at
            # position 0, which reads as a truncated file and ends the iteration
            # dozens of records into a 21,172-record corpus.
            buf = buf[1:]
            expect_separator = False
            continue

        try:
            element, end = decoder.raw_decode(buf)
        except ValueError as exc:
            # Either the element straddles the buffer boundary, or the input is
            # genuinely malformed. They are indistinguishable here, so extend the
            # buffer and retry; only end-of-input settles it.
            chunk = fp.read(buffer_size)
            if not chunk:
                raise ToolkitError(f"malformed or truncated JSON array: {exc}") from exc
            buf += chunk
            continue

        yield element
        buf = buf[end:]
        expect_separator = True
