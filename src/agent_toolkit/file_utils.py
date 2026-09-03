"""adapter. Files on disk to values, and back atomically."""

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
    try:
        with open(path, encoding="utf-8-sig") as fp:
            return fp.read()
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("read_txt(%s) failed: %s", path, exc)
        return ""


def read_json(path: str | os.PathLike[str]) -> Any:
    try:
        with open(path, encoding="utf-8-sig") as fp:
            return json.load(fp)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("read_json(%s) failed: %s", path, exc)
        return {}


def read_yaml(path: str | os.PathLike[str]) -> Any:
    try:
        with open(path, encoding="utf-8-sig") as fp:
            loaded = yaml.safe_load(fp)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        logger.debug("read_yaml(%s) failed: %s", path, exc)
        return {}
    return {} if loaded is None else loaded


def read_jsonlines(path: str | os.PathLike[str]) -> list[Any]:
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
    with open(path, encoding=encoding) as fp:
        yield from iter_json_array(fp, buffer_size=buffer_size)


@contextmanager
def _atomic_write(path: str | os.PathLike[str]) -> Iterator[IO[str]]:
    destination = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(dir=parent, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fp:
            yield fp
        os.replace(temp_path, destination)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def write_json(path: str | os.PathLike[str], data: Any, *, indent: int = 2) -> None:
    with _atomic_write(path) as fp:
        json.dump(data, fp, indent=indent, ensure_ascii=False)


def write_jsonlines(path: str | os.PathLike[str], rows: Iterable[Any]) -> None:
    with _atomic_write(path) as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")
