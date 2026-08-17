#!/bin/bash
# Local development setup using uv.
set -euo pipefail

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies (including the llm extra)..."
uv sync --extra llm

echo "Installing pre-commit hooks..."
uv run pre-commit install

echo
echo "Done. Checks:"
echo "  uv run ruff check . && uv run ruff format --check ."
echo "  uv run mypy --strict src/agent_toolkit"
echo "  uv run pytest -q"
