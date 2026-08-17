"""Error base class.

Every exception this library raises derives from :class:`ToolkitError`, so a
caller can catch everything from the toolkit with one ``except`` clause without
also catching its own bugs.
"""

__all__ = ["ToolkitError"]


class ToolkitError(Exception):
    """Base class for every error raised by agent_toolkit."""
