"""facade. The embeddings route's surface, gated on the 'llm' extra."""

try:
    import openai
except ImportError as exc:
    raise ImportError(
        "agent_toolkit.embed needs the optional 'llm' extra: "
        "pip install 'agent-toolkit[llm]'"
    ) from exc

from agent_toolkit.embed.config import (
    ConfigResolver,
    DictConfigResolver,
    EmbedConfig,
    EnvConfigResolver,
    JsonDirConfigResolver,
    YamlConfigResolver,
    resolve_config,
    set_config_resolver,
)
from agent_toolkit.embed.factory import (
    DEFAULT_EXPONENTIAL_BACKOFF,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    vectors,
)
from agent_toolkit.embed.traffic_control import (
    TrafficController,
    get_traffic_controller,
)
from agent_toolkit.logging import configure_logging, get_logger

__all__ = [
    "DEFAULT_EXPONENTIAL_BACKOFF",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "ConfigResolver",
    "DictConfigResolver",
    "EmbedConfig",
    "EnvConfigResolver",
    "JsonDirConfigResolver",
    "TrafficController",
    "YamlConfigResolver",
    "configure_logging",
    "vectors",
    "get_logger",
    "get_traffic_controller",
    "resolve_config",
    "set_config_resolver",
]
