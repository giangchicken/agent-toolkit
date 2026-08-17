"""Shared utilities for agent and dataset pipelines.

This package configures no logging handlers and reads no environment variable at
import time; see :mod:`agent_toolkit.logging`. The LLM client lives behind the
``agent-toolkit[llm]`` extra and is deliberately not imported from here, so
``import agent_toolkit`` never pulls the OpenAI SDK.
"""

from agent_toolkit.errors import ToolkitError
from agent_toolkit.logging import get_logger

__version__ = "0.1.0"

__all__ = ["ToolkitError", "__version__", "get_logger"]
