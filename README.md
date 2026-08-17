# agent-toolkit

Shared utilities for agent and dataset pipelines. Spec: [`../docs/agent-toolkit/spec.md`](../docs/agent-toolkit/spec.md). Plan: [`../docs/agent-toolkit/plan.md`](../docs/agent-toolkit/plan.md).

Two guarantees shape the design:

- **The core is light.** `string_utils`, `json_utils`, and `file_utils` depend on `json-repair` and `pyyaml` only. Importing them does not import the OpenAI SDK.
- **The library configures nothing.** No logging handler, no logger level, no environment variable read at import time. The host owns all of that.

## Install

```bash
pip install agent-toolkit           # core
pip install "agent-toolkit[llm]"    # core + LLM client
```

## Develop

```bash
./setup-dev-uv.sh
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/agent_toolkit
uv run pytest -q
```

## Status

v0.1.0 is under construction; see the plan for task order. The public surface and the deferred v0.2 backlog are documented in T11.
