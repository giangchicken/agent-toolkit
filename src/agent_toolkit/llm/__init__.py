"""facade. The chat route's surface, gated on the 'llm' extra."""

try:
    import openai
except ImportError as exc:
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
    YamlConfigResolver,
    resolve_config,
    set_config_resolver,
)
from agent_toolkit.llm.executors import Completion
from agent_toolkit.llm.factory import (
    DEFAULT_EXPONENTIAL_BACKOFF,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    complete,
    complete_with_reasoning,
)
from agent_toolkit.llm.model_meta import (
    count_tokens,
    model_family,
    supports_native_tool_calling,
    supports_reasoning,
)
from agent_toolkit.llm.traffic_control import TrafficController, get_traffic_controller

__all__ = [
    "DEFAULT_EXPONENTIAL_BACKOFF",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "Completion",
    "ConfigResolver",
    "DictConfigResolver",
    "EnvConfigResolver",
    "JsonDirConfigResolver",
    "LLMConfig",
    "TrafficController",
    "YamlConfigResolver",
    "complete",
    "complete_with_reasoning",
    "count_tokens",
    "get_traffic_controller",
    "model_family",
    "resolve_config",
    "set_config_resolver",
    "supports_native_tool_calling",
    "supports_reasoning",
]
