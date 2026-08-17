"""File readers and writers for pipeline artifacts.

Harvested from ``voice-agent-toolkit``'s ``file_utils``, minus its ``jsonlines``
dependency: one JSON value per line is four lines of stdlib, and the core
dependency list is a guarantee this library makes (requirement 3).

Readers and writers fail differently, deliberately:

- **Readers return a default** -- ``""``, ``{}``, ``[]`` -- and log at debug.
  This is the harvested contract, and it is load-bearing:
  ``agent-evaluation``'s ``get_llm_config`` branches on ``if not raw:`` after
  calling ``read_json``, so a reader that raised would turn a missing config
  file from a logged warning into an unhandled exception, and requirement 6
  promises migration is an import-line change and nothing else.
- **Writers raise.** They have no return value a caller could inspect, so a
  swallowed failure means a pipeline stage records an artifact it never wrote.
  That is the one failure mode worse than a crash.

Every reader opens with ``utf-8-sig``, which decodes plain UTF-8 unchanged but
also strips a byte-order mark if one is present. Writers use ``utf-8`` and never
emit one. Without this, a BOM'd file would reach ``json.load`` as an unexpected
first character and -- given the reader contract above -- be reported as an
empty config rather than a broken one.
"""

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import IO, Any

import yaml

from agent_toolkit.json_utils import DEFAULT_BUFFER_SIZE, iter_json_array
from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "iter_json_array_file",
    "read_json",
    "read_jsonlines",
    "read_txt",
    "read_yaml",
    "write_json",
    "write_jsonlines",
]


def read_txt(path: str | os.PathLike[str]) -> str:
    """Return the file's text, or ``""`` if it cannot be read."""
    try:
        with open(path, encoding="utf-8-sig") as fp:
            return fp.read()
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("read_txt(%s) failed: %s", path, exc)
        return ""


def read_json(path: str | os.PathLike[str]) -> Any:
    """Return the parsed JSON, or ``{}`` if the file is missing or malformed.

    Typed ``Any`` rather than ``dict`` because a JSON document's top level can
    be any value and this function reports what it finds. For a top-level array
    too large to hold in memory, use :func:`iter_json_array_file`.
    """
    try:
        with open(path, encoding="utf-8-sig") as fp:
            return json.load(fp)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("read_json(%s) failed: %s", path, exc)
        return {}


def read_yaml(path: str | os.PathLike[str]) -> Any:
    """Return the parsed YAML, or ``{}`` if the file is missing or malformed.

    ``safe_load``, not ``load``: a pipeline reads parameter files, and YAML's
    full loader can construct arbitrary Python objects from them.

    An empty file yields ``{}`` rather than ``None``, so callers need only the
    one falsy case the other readers give them.
    """
    try:
        with open(path, encoding="utf-8-sig") as fp:
            loaded = yaml.safe_load(fp)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.debug("read_yaml(%s) failed: %s", path, exc)
        return {}
    return {} if loaded is None else loaded


def read_jsonlines(path: str | os.PathLike[str]) -> list[Any]:
    """Return one parsed JSON value per non-blank line, or ``[]`` on failure.

    A malformed line discards the whole read rather than returning the rows
    before it: a partial artifact that looks complete is the failure this
    library's streaming reader also exists to prevent.
    """
    rows: list[Any] = []
    try:
        with open(path, encoding="utf-8-sig") as fp:
            for line in fp:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("read_jsonlines(%s) failed: %s", path, exc)
        return []
    return rows


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


@contextmanager
def _atomic_write(path: str | os.PathLike[str]) -> Iterator[IO[str]]:
    """Yield a writable handle whose contents replace ``path`` only on success.

    The temp file is created in the *destination directory* so ``os.replace`` is
    a same-filesystem rename, which is atomic. A temp file under ``/tmp`` would
    make it a copy across devices and reopen the partial-write window this
    exists to close.

    Durability is not promised: there is no ``fsync``, so a power loss can still
    lose the write. What is promised is that a crash -- or a serialization error
    partway through -- leaves the previous file exactly as it was, which is the
    failure requirement 8 names.
    """
    destination = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fp:
            yield fp
        os.replace(temp_path, destination)
    finally:
        # A successful replace renamed the temp file away, so this only fires on
        # the failure path.
        if os.path.exists(temp_path):
            os.remove(temp_path)


def write_json(path: str | os.PathLike[str], data: Any, *, indent: int = 2) -> None:
    """Write ``data`` as JSON to ``path``, atomically, creating parent dirs.

    ``ensure_ascii=False``, so Vietnamese text stays readable in the artifact
    rather than becoming ``\\uXXXX`` escapes.

    The harvested version's ``merge_if_exist`` is dropped: read-modify-write is
    the caller's decision and, done here, races with any concurrent writer.
    """
    with _atomic_write(path) as fp:
        json.dump(data, fp, indent=indent, ensure_ascii=False)


def write_jsonlines(path: str | os.PathLike[str], rows: Iterable[Any]) -> None:
    """Write one JSON value per line to ``path``, atomically.

    Takes any iterable, so a generator streams to disk without the caller
    materializing every row first.

    The harvested version's ``mode="a"`` is dropped: no v0.1 caller appends, and
    an append cannot be made atomic by replace-on-success.
    """
    with _atomic_write(path) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")
