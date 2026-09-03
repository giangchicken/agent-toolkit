"""facade. The core surface: errors and logger access."""

from agent_toolkit.errors import ToolkitError
from agent_toolkit.logging import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = ["ToolkitError", "__version__", "configure_logging", "get_logger"]
