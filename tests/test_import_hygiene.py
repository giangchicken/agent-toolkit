"""The structural guarantees every other module depends on.

These tests exist because the toolkit this library harvests from gets both of
them wrong: its package ``__init__`` attaches a StreamHandler, reads an
environment variable, and calls ``setLevel`` at import time, and it declares
seventeen hard dependencies with no extras, so importing one string helper pulls
in pandas, boto3, elasticsearch, kafka, redis, sqlalchemy, and tiktoken.

Spec requirements 4 and 5; invariants 1, 2, and 5.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

import agent_toolkit

PACKAGE_ROOT = pathlib.Path(agent_toolkit.__file__).parent
SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))


def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_source_files_were_found() -> None:
    """Guard against the rglob silently matching nothing."""
    assert len(SOURCE_FILES) >= 6


# --- invariant 5: no silent logging config -----------------------------------

_LOGGING_PROBE = """
import importlib
import logging
import pkgutil

root = logging.getLogger()
pkg = logging.getLogger("agent_toolkit")
before = (len(root.handlers), root.level, len(pkg.handlers), pkg.level)

import agent_toolkit

for module in pkgutil.walk_packages(agent_toolkit.__path__, "agent_toolkit."):
    try:
        importlib.import_module(module.name)
    except ImportError:
        pass  # an optional extra is not installed; nothing to check

after = (len(root.handlers), root.level, len(pkg.handlers), pkg.level)
assert before == after, f"import changed logging state: {before} -> {after}"
print("ok")
"""


def test_importing_every_module_leaves_logging_untouched() -> None:
    """No handler is added and no level is changed, on the root or our logger.

    Runs in a subprocess because pytest installs its own handlers on the root
    logger, which would make an in-process assertion measure pytest, not us.
    """
    assert _run(_LOGGING_PROBE) == "ok"


def test_get_logger_returns_the_stdlib_logger_unmodified() -> None:
    import logging

    logger = agent_toolkit.get_logger("agent_toolkit.probe")
    assert logger is logging.getLogger("agent_toolkit.probe")
    assert logger.handlers == []
    assert logger.level == logging.NOTSET


# --- invariant 2: core stays light ------------------------------------------

_CORE_IMPORT_PROBE = """
import sys

import agent_toolkit
import agent_toolkit.file_utils
import agent_toolkit.json_utils
import agent_toolkit.string_utils

leaked = sorted(
    name
    for name in ("openai", "tenacity", "aiohttp", "tiktoken")
    if name in sys.modules
)
assert not leaked, f"core import pulled the llm extra: {leaked}"
print("ok")
"""


def test_core_import_does_not_pull_the_llm_extra() -> None:
    assert _run(_CORE_IMPORT_PROBE) == "ok"


_EXTRA_GATE_PROBE = """
import sys

# A None entry in sys.modules makes `import openai` raise ImportError, which is
# what a core-only install looks like from inside the interpreter.
sys.modules["openai"] = None

try:
    import agent_toolkit.llm
except ImportError as exc:
    assert "llm" in str(exc).lower(), f"the error does not name the extra: {exc}"
    print("ok")
else:
    raise SystemExit("importing agent_toolkit.llm without openai should have failed")
"""


def test_the_llm_subpackage_names_its_extra_when_openai_is_missing() -> None:
    """The other half of invariant 2: the optional half says it is optional.

    A bare ``No module named 'openai'`` from inside a config module tells a
    caller nothing about what to install. T11 checks this against a real
    core-only wheel; this checks the gate itself.
    """
    assert _run(_EXTRA_GATE_PROBE) == "ok"


# --- requirement 4: no environment read at import time -----------------------


def _import_time_expressions(tree: ast.Module) -> list[ast.expr]:
    """Every expression that runs when the module is imported.

    Function *bodies* are excluded because they run when called, not when
    imported. Everything else is included, deliberately: class bodies, default
    arguments, and decorators all execute at import time, and each is a real
    place an environment read could hide.
    """
    found: list[ast.expr] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.expr):
            found.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for default in child.args.defaults:
                    visit(default)
                for kw_default in child.args.kw_defaults:
                    if kw_default is not None:
                        visit(kw_default)
                if not isinstance(child, ast.Lambda):
                    for decorator in child.decorator_list:
                        visit(decorator)
                continue
            visit(child)

    visit(tree)
    return found


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_environment_read_at_import_time(path: pathlib.Path) -> None:
    """Reading the environment inside a function is fine; at import time is not.

    A caller who imports this library must not have its behavior decided by
    whatever happened to be exported in their shell.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        ast.unparse(node)
        for node in _import_time_expressions(tree)
        if isinstance(node, (ast.Call, ast.Subscript))
        and ("getenv" in ast.unparse(node) or "environ" in ast.unparse(node))
    ]
    assert not offenders, f"{path.name} reads the environment at import: {offenders}"


def test_the_environment_check_would_catch_a_violation() -> None:
    """The check above passes trivially today; prove it is not vacuous."""
    violation = ast.parse("import os\nDEBUG = os.getenv('DEBUG_TOOLKIT')\n")
    offenders = [
        node
        for node in _import_time_expressions(violation)
        if isinstance(node, (ast.Call, ast.Subscript)) and "getenv" in ast.unparse(node)
    ]
    assert offenders, "the import-time environment check is vacuous"

    innocent = ast.parse("import os\ndef f():\n    return os.getenv('X')\n")
    assert not [
        node
        for node in _import_time_expressions(innocent)
        if isinstance(node, (ast.Call, ast.Subscript)) and "getenv" in ast.unparse(node)
    ], "the check wrongly flags a runtime environment read"


# --- invariant 1 and requirement 5: no host coupling -------------------------


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_import_from_a_host_application(path: pathlib.Path) -> None:
    """``from src.…`` is how the harvested code reached into its host."""
    source = path.read_text(encoding="utf-8")
    offenders = [
        f"{path.name}:{n}: {line.strip()}"
        for n, line in enumerate(source.splitlines(), 1)
        if line.lstrip().startswith(("from src.", "import src."))
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_path_relative_to_the_package_itself(path: pathlib.Path) -> None:
    """The strict form of the spec's ``Path(__file__).*parent.*configs`` check.

    The library ships no data files, so it has no legitimate reason to resolve
    anything against its own location. Banning ``__file__`` outright is simpler
    than matching the specific four-parents-up-to-``configs`` shape, and it
    forecloses the variants of that shape too.
    """
    source = path.read_text(encoding="utf-8")
    offenders = [
        f"{path.name}:{n}: {line.strip()}"
        for n, line in enumerate(source.splitlines(), 1)
        if "__file__" in line
    ]
    assert not offenders, offenders


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_no_host_config_directory_is_named(path: pathlib.Path) -> None:
    """Requirement 5: no module may reference a ``configs/`` or ``prompts/`` path."""
    source = path.read_text(encoding="utf-8")
    banned = ('"configs', "'configs", "configs/", '"prompts', "'prompts", "prompts/")
    offenders = [
        f"{path.name}:{n}: {line.strip()}"
        for n, line in enumerate(source.splitlines(), 1)
        if any(token in line for token in banned)
    ]
    assert not offenders, offenders
