"""adapter. Domain records to a console and a daily file."""

import logging
import pathlib
import sys
from datetime import datetime
from typing import IO, Any

__all__ = ["configure_logging", "get_logger"]

_ROOT_NAME = "agent_toolkit"
_INSTALLED = "_agent_toolkit_installed"

_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[37m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"

_FILE_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, colors: bool) -> None:
        super().__init__(datefmt=_DATE_FORMAT)
        self._colors = colors

    def format(self, record: logging.LogRecord) -> str:
        module = record.name
        prefix = _ROOT_NAME + "."
        if module.startswith(prefix):
            module = module[len(prefix) :]
        if self._colors:
            color = _COLORS.get(record.levelname, "")
            head = f"{_DIM}[{module}]{_RESET} {color}{record.levelname}:{_RESET}"
        else:
            head = f"[{module}] {record.levelname}:"
        text = f"{head} {record.getMessage()}"
        if record.exc_info:
            text = f"{text}\n{self.formatException(record.exc_info)}"
        return text


def _as_level(value: int | str) -> int:
    if isinstance(value, int):
        return value
    resolved = logging.getLevelNamesMapping().get(value.upper())
    if resolved is None:
        raise ValueError(f"unknown log level {value!r}")
    return resolved


def _is_tty(stream: IO[Any]) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _uninstall(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _INSTALLED, False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    level: int | str = "INFO",
    *,
    console: bool = True,
    log_dir: str | pathlib.Path | None = None,
    file_level: int | str = "DEBUG",
    colors: bool | None = None,
    filename_prefix: str = "agent_toolkit",
    propagate: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(_ROOT_NAME)
    _uninstall(logger)

    console_level = _as_level(level)
    levels = [console_level]

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(console_level)
        stream.setFormatter(
            _ConsoleFormatter(colors if colors is not None else _is_tty(sys.stderr))
        )
        setattr(stream, _INSTALLED, True)
        logger.addHandler(stream)

    if log_dir is not None:
        directory = pathlib.Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        handler = logging.FileHandler(
            directory / f"{filename_prefix}_{stamp}.log", encoding="utf-8"
        )
        resolved_file_level = _as_level(file_level)
        handler.setLevel(resolved_file_level)
        handler.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
        setattr(handler, _INSTALLED, True)
        logger.addHandler(handler)
        levels.append(resolved_file_level)

    logger.setLevel(min(levels))
    logger.propagate = propagate
    return logger
