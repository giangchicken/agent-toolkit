"""adapter. Env, JSON and YAML translated into one embeddings call's settings."""

import os
import pathlib
from dataclasses import dataclass, fields, replace
from typing import Any, Protocol

from agent_toolkit.embed.exceptions import EmbedConfigError
from agent_toolkit.file_utils import read_json, read_yaml
from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "ConfigResolver",
    "DictConfigResolver",
    "EmbedConfig",
    "EnvConfigResolver",
    "JsonDirConfigResolver",
    "YamlConfigResolver",
    "resolve_config",
    "set_config_resolver",
]


@dataclass
class EmbedConfig:
    model: str
    api_key: str = ""
    base_url: str | None = None
    binding: str = "openai"
    extra_headers: dict[str, str] | None = None
    timeout: float = 120.0
    max_concurrency: int = 20
    requests_per_minute: int = 600


_FIELD_NAMES = frozenset(field.name for field in fields(EmbedConfig))

_CALLER_SUPPLIED = ("api_key", "base_url")

_YAML_KEYS = _FIELD_NAMES - set(_CALLER_SUPPLIED)


class ConfigResolver(Protocol):
    def resolve(self, model: str | None) -> EmbedConfig: ...


class EnvConfigResolver:
    def resolve(self, model: str | None) -> EmbedConfig:
        resolved = model or os.environ.get("EMBED_MODEL") or ""
        if not resolved:
            raise EmbedConfigError(
                "no embeddings model: pass model=..., set EMBED_MODEL, "
                "or install a resolver with agent_toolkit.embed.set_config_resolver()"
            )
        return EmbedConfig(
            model=resolved,
            api_key=os.environ.get("EMBED_API_KEY", ""),
            base_url=os.environ.get("EMBED_BASE_URL"),
        )


class DictConfigResolver:
    def __init__(self, configs: dict[str, EmbedConfig]) -> None:
        self._configs = dict(configs)

    def resolve(self, model: str | None) -> EmbedConfig:
        if model is None:
            raise EmbedConfigError(
                f"a model name is required; known models: {sorted(self._configs)}"
            )
        try:
            return self._configs[model]
        except KeyError:
            raise EmbedConfigError(
                f"no config for model {model!r}; known models: {sorted(self._configs)}"
            ) from None


class JsonDirConfigResolver:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = pathlib.Path(directory)

    def resolve(self, model: str | None) -> EmbedConfig:
        if model is None:
            raise EmbedConfigError("a model name is required to choose a config file")
        name = model.strip()
        path = self._directory / (name.lower().replace(" ", "_") + ".json")
        raw = read_json(path)
        if not isinstance(raw, dict) or not raw:
            raise EmbedConfigError(f"no usable embeddings config at {path}")

        known: dict[str, Any] = {
            key: value for key, value in raw.items() if key in _FIELD_NAMES
        }
        known.setdefault("model", name)
        return EmbedConfig(**known)


class YamlConfigResolver(DictConfigResolver):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        raw = read_yaml(path)
        if not isinstance(raw, dict) or not raw:
            raise EmbedConfigError(f"no usable embeddings config at {path}")

        defaults = raw.get("defaults") or {}
        if not isinstance(defaults, dict):
            raise EmbedConfigError(f"{path}: 'defaults' must be a mapping")
        _reject_unsettable(defaults, path=path, where="defaults")

        models = raw.get("models")
        if not isinstance(models, dict) or not models:
            raise EmbedConfigError(f"{path}: no 'models' block, or it is empty")

        configs: dict[str, EmbedConfig] = {}
        for name, settings in models.items():
            if settings is None:
                settings = {}
            if not isinstance(settings, dict):
                raise EmbedConfigError(f"{path}: model {name!r} must be a mapping")
            _reject_unsettable(settings, path=path, where=f"model {name!r}")
            merged: dict[str, Any] = {**defaults, **settings}
            merged.setdefault("model", name)
            try:
                configs[str(name)] = EmbedConfig(**merged)
            except TypeError as exc:
                raise EmbedConfigError(f"{path}: model {name!r}: {exc}") from exc

        super().__init__(configs)


def _reject_unsettable(settings: dict[Any, Any], *, path: Any, where: str) -> None:
    for key in _CALLER_SUPPLIED:
        if key in settings:
            raise EmbedConfigError(
                f"{path}: {where} sets {key!r}; this file says how a model "
                f"behaves, not where it runs or what authenticates it. "
                f"Pass {key}=... to vectors() or resolve_config()"
            )
    unknown = sorted(str(key) for key in settings if key not in _YAML_KEYS)
    if unknown:
        raise EmbedConfigError(
            f"{path}: {where} has unknown setting(s) {unknown}; "
            f"settable: {sorted(_YAML_KEYS)}"
        )


_resolver: ConfigResolver | None = None


def set_config_resolver(resolver: ConfigResolver | None) -> None:
    global _resolver
    _resolver = resolver


def resolve_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    binding: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> EmbedConfig:
    resolver = _resolver if _resolver is not None else EnvConfigResolver()
    config = resolver.resolve(model)
    logger.debug(
        "%s resolved %r to model=%s base_url=%s api_key=%s",
        type(resolver).__name__,
        model,
        config.model,
        config.base_url,
        "set" if config.api_key else "blank",
    )

    headers = dict(config.extra_headers or {})
    if extra_headers:
        headers.update(extra_headers)

    return replace(
        config,
        api_key=api_key if api_key is not None else config.api_key,
        base_url=base_url or config.base_url,
        binding=binding or config.binding or "openai",
        extra_headers=headers,
    )
