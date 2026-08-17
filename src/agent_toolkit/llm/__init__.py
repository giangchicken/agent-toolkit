"""OpenAI-compatible LLM client. Requires the ``agent-toolkit[llm]`` extra.

The gate below is what keeps the core light: importing this subpackage without
the extra installed must say so, rather than failing later with a bare
``No module named 'openai'`` from three frames down.

``stream`` and ``complete_with_tools`` are v0.2; only ``complete`` ships in v0.1.
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
from agent_toolkit.llm.factory import complete
from agent_toolkit.llm.retry import (
    RetryPolicy,
    get_default_retry_policy,
    set_default_retry_policy,
)
from agent_toolkit.llm.traffic_control import TrafficController, get_traffic_controller

__all__ = [
    "ConfigResolver",
    "DictConfigResolver",
    "EnvConfigResolver",
    "JsonDirConfigResolver",
    "LLMConfig",
    "RetryPolicy",
    "TrafficController",
    "complete",
    "get_default_retry_policy",
    "get_traffic_controller",
    "resolve_config",
    "set_config_resolver",
    "set_default_retry_policy",
]
