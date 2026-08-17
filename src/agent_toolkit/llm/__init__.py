"""OpenAI-compatible LLM client. Requires the ``agent-toolkit[llm]`` extra.

The gate below is what keeps the core light: importing this subpackage without
the extra installed must say so, rather than failing later with a bare
``No module named 'openai'`` from three frames down.

``complete``, ``RetryPolicy``, and ``TrafficController`` join this surface in
T7 and T8.
"""

try:
    import openai  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised by the T11 wheel check
    raise ImportError(
        "agent_toolkit.llm needs the optional 'llm' extra: "
        "pip install 'agent-toolkit[llm]'"
    ) from exc

from agent_toolkit.llm.config import (
    ConfigResolver,
    DictConfigResolver,
    EnvConfigResolver,
    JsonDirConfigResolver,
    LLMConfig,
    resolve_config,
    set_config_resolver,
)

__all__ = [
    "ConfigResolver",
    "DictConfigResolver",
    "EnvConfigResolver",
    "JsonDirConfigResolver",
    "LLMConfig",
    "resolve_config",
    "set_config_resolver",
]
