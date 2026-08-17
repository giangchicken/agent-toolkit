"""Logger access.

This module deliberately does nothing but hand back a standard library logger.
It attaches no handler, sets no level, and installs no formatter, because a
library that configures logging steals a decision that belongs to the host
application. Configuring the ``agent_toolkit`` logger is the host's job.
"""

import logging

__all__ = ["get_logger"]


def get_logger(name: str) -> logging.Logger:
    """Return the standard library logger called ``name``.

    Call it as ``get_logger(__name__)``. No side effects.
    """
    return logging.getLogger(name)
