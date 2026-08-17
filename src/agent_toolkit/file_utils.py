"""File readers and writers.

T4 adds only the streaming-array reader, which is the wrapper its criteria place
here rather than in :mod:`agent_toolkit.json_utils`. The rest of this module --
``read_txt``, ``read_json``, ``read_yaml``, atomic ``write_json``, and the JSONL
pair -- arrives in T5.
"""

import os
from collections.abc import Iterator
from typing import Any

from agent_toolkit.json_utils import DEFAULT_BUFFER_SIZE, iter_json_array

__all__ = ["iter_json_array_file"]


def iter_json_array_file(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8-sig",
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> Iterator[Any]:
    """Stream a top-level JSON array from the file at ``path``.

    Defaults to ``utf-8-sig``, which decodes plain UTF-8 unchanged but also
    strips a byte-order mark if one is present. A BOM is not whitespace, so
    under plain ``utf-8`` it would reach the parser as the array's first
    character and the file would be reported as not being an array at all.
    """
    with open(path, encoding=encoding) as fp:
        yield from iter_json_array(fp, buffer_size=buffer_size)
