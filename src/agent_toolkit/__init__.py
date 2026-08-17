"""Shared utilities for agent and dataset pipelines.

This package configures no logging handlers and reads no environment variable at
import time; see ``agent_toolkit.logging``. The LLM client lives behind the
``agent-toolkit[llm]`` extra and is not imported from here.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
