"""The package imports and reports a version."""

import agent_toolkit


def test_version_is_exposed() -> None:
    assert agent_toolkit.__version__ == "0.1.0"
