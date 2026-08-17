"""Where LLM settings come from, without reaching into a host directory.

The harvested module computed a directory four parents above its own file and
read ``<model>.json`` from it. That is the coupling that makes the original
unliftable: a library cannot assume anything about the layout of the application
that installed it.

What replaces it is a resolver the host supplies once at startup:

- :class:`EnvConfigResolver` -- the default. ``LLM_MODEL``, ``LLM_API_KEY``,
  ``LLM_BASE_URL``, read when ``resolve`` is called, never at import.
- :class:`DictConfigResolver` -- an in-memory mapping, for tests and for a
  pipeline that keeps its model settings in its own parameter file or database.
- :class:`JsonDirConfigResolver` -- a directory of ``<model>.json`` files,
  preserving the harvested convention so ``agent-evaluation`` keeps working by
  installing one at startup.

Precedence, and it is exactly two levels deep: an explicit argument to
:func:`resolve_config` beats whatever the installed resolver returns, and the
environment resolver is what "installed resolver" means when nothing was
installed. The environment is a *default resolver*, not a layer underneath the
others -- a stray ``LLM_API_KEY`` in a developer's shell must not seep into a
pipeline run that installed a resolver of its own.

The model name is never guessed. The harvested code defaulted to a specific
model of its host's choosing; a library that does that is choosing for its
callers.
"""

import os
import pathlib
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

from agent_toolkit.file_utils import read_json
from agent_toolkit.llm.exceptions import LLMConfigError

__all__ = [
    "ConfigResolver",
    "DictConfigResolver",
    "EnvConfigResolver",
    "JsonDirConfigResolver",
    "LLMConfig",
    "resolve_config",
    "set_config_resolver",
]


@dataclass
class LLMConfig:
    """Everything one OpenAI-compatible endpoint needs to be called.

    Four fields of the harvested dataclass are gone, each with no reader:
    ``effective_url`` and ``provider_mode`` (nothing outside the config module
    ever read them), ``provider_name`` (never set by the loader, never read by
    the caller -- the traffic controller takes its own), and
    ``traffic_controller``, whose construction inside config loading is the
    defect T7 exists to fix.
    """

    model: str
    api_key: str = ""
    base_url: str | None = None
    api_version: str | None = None
    binding: str = "openai"
    extra_headers: dict[str, str] | None = None
    reasoning_effort: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    max_concurrency: int = 20
    requests_per_minute: int = 600


_FIELD_NAMES = frozenset(field.name for field in fields(LLMConfig))


class ConfigResolver(Protocol):
    def resolve(self, model: str | None) -> LLMConfig: ...


class EnvConfigResolver:
    """Reads ``LLM_MODEL``, ``LLM_API_KEY``, and ``LLM_BASE_URL``.

    Only those three. Every other field takes its dataclass default; a host that
    needs to set more installs a :class:`DictConfigResolver` instead of growing
    an environment-variable vocabulary this library would then have to own.
    """

    def resolve(self, model: str | None) -> LLMConfig:
        resolved = model or os.environ.get("LLM_MODEL") or ""
        if not resolved:
            raise LLMConfigError(
                "no model: pass model=..., set LLM_MODEL, "
                "or install a resolver with set_config_resolver()"
            )
        return LLMConfig(
            model=resolved,
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL"),
        )


class DictConfigResolver:
    """Resolves against an in-memory mapping of model name to config."""

    def __init__(self, configs: dict[str, LLMConfig]) -> None:
        self._configs = dict(configs)

    def resolve(self, model: str | None) -> LLMConfig:
        if model is None:
            raise LLMConfigError(
                f"a model name is required; known models: {sorted(self._configs)}"
            )
        try:
            return self._configs[model]
        except KeyError:
            raise LLMConfigError(
                f"no config for model {model!r}; known models: {sorted(self._configs)}"
            ) from None


class JsonDirConfigResolver:
    """Resolves ``<model>.json`` from one directory the host names.

    The filename is the harvested convention exactly: lowercased, spaces to
    underscores, so ``"GLM-5.1"`` reads ``glm-5.1.json`` and ``"My Model"`` reads
    ``my_model.json``.

    Unlike the harvested loader, a missing or unreadable file raises rather than
    returning an empty config. The original logged "API key will be blank,
    requests will likely fail" and carried on -- a warning conceding that the
    call it is about to make cannot work. :class:`LLMConfigError` exists for
    precisely this, and raising it here reports the problem at the place that
    can name the file.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = pathlib.Path(directory)

    def resolve(self, model: str | None) -> LLMConfig:
        if model is None:
            raise LLMConfigError("a model name is required to choose a config file")
        name = model.strip()
        path = self._directory / (name.lower().replace(" ", "_") + ".json")
        raw = read_json(path)
        if not isinstance(raw, dict) or not raw:
            raise LLMConfigError(f"no usable LLM config at {path}")

        # Unknown keys are ignored rather than fatal: these files are written by
        # hand and outlive the field set of any one library version.
        known: dict[str, Any] = {
            key: value for key, value in raw.items() if key in _FIELD_NAMES
        }
        known.setdefault("model", name)
        return LLMConfig(**known)


_resolver: ConfigResolver | None = None


def set_config_resolver(resolver: ConfigResolver | None) -> None:
    """Install the process-wide resolver. ``None`` restores the environment one."""
    global _resolver
    _resolver = resolver


def resolve_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str | None = None,
    extra_headers: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
) -> LLMConfig:
    """Resolve one call's settings: explicit arguments over the resolver's answer.

    ``model`` is the exception, because it plays two roles: it is also the key
    the resolver looks up. So the resolver decides what the selected config's
    ``model`` field says -- the environment resolver echoes the argument, a
    directory resolver prefers the ``model`` recorded inside the file it read.

    ``extra_headers`` merges rather than replaces, so a caller can add one header
    without discarding the configured ones. The merge is into a *copy*: the
    harvested version updated the resolved config's own dict in place, and since
    that config came from a process-wide cache, one call's header became every
    later call's header.
    """
    resolver = _resolver if _resolver is not None else EnvConfigResolver()
    config = resolver.resolve(model)

    headers = dict(config.extra_headers or {})
    if extra_headers:
        headers.update(extra_headers)

    return replace(
        config,
        api_key=api_key if api_key is not None else config.api_key,
        base_url=base_url or config.base_url,
        api_version=api_version or config.api_version,
        binding=binding or config.binding or "openai",
        extra_headers=headers,
        reasoning_effort=(
            reasoning_effort
            if reasoning_effort is not None
            else config.reasoning_effort
        ),
    )
