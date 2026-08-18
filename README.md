# agent-toolkit

Shared utilities for agent and dataset pipelines: string, JSON, and file helpers, plus an OpenAI-compatible LLM client with retry, rate limiting, and validated structured output.

Spec: [`../docs/agent-toolkit/spec.md`](../docs/agent-toolkit/spec.md). Plan: [`../docs/agent-toolkit/plan.md`](../docs/agent-toolkit/plan.md).

Two guarantees shape the design:

- **The core is light.** `string_utils`, `json_utils`, and `file_utils` depend on `json-repair` and `pyyaml` only. Importing them does not import the OpenAI SDK.
- **The library configures nothing.** No logging handler, no logger level, no environment variable read at import time. The host owns all of that.

## Install

```bash
pip install agent-toolkit           # core
pip install "agent-toolkit[llm]"    # core + LLM client
```

`agent-toolkit` is not on a registry yet, so an installed copy comes from a path or a git ref. It lives in a subdirectory of the DataForce monorepo, which both forms have to name:

```bash
pip install "agent-toolkit[llm] @ git+https://github.com/giangchicken/DataForce.git@agent-toolkit-v0.1.0#subdirectory=agent-toolkit"
pip install "/path/to/DataForce/agent-toolkit[llm]"     # editable: add -e
```

Python 3.11 or newer. Importing `agent_toolkit.llm` without the `llm` extra raises an `ImportError` naming the extra rather than failing later with `No module named 'openai'`.

## string_utils

```python
from agent_toolkit.string_utils import (
    compute_hash,
    extract_json_from_text,
    normalize_text,
    slot_filling,
    split_thinking,
)

slot_filling("Xin chào {{name}}", {"name": "Dũng"})  # 'Xin chào Dũng'
normalize_text("  Xin  chào  ")  # 'Xin chào'
normalize_text("Xin chào", remove_tone_marks=True)  # 'Xin chao'
compute_hash("dataforce")  # sha256 hex digest

# Parses what a model actually returns: fences, prose around it, minor malformation.
extract_json_from_text('Vote:\n```json\n["get_weather"]\n```')  # ['get_weather']

# Reasoning and answer, separated. Handles a chat template that pre-filled the
# opening tag and a block that max_tokens cut off, which clean_thinking_tags does not.
split_thinking("Chào hỏi thôi.</think>Xin chào!")
# ('Chào hỏi thôi.</think>', 'Xin chào!')
```

`clean_thinking_tags` is also exported, unchanged from `voice-agent-toolkit`, for call sites being migrated. New code should use `split_thinking`.

## json_utils and file_utils

`iter_json_array` streams a top-level JSON array element by element, so a 126 MiB corpus never loads whole. Memory is bounded by the buffer plus the largest single element.

```python
from agent_toolkit.file_utils import (
    iter_json_array_file,
    read_json,
    read_jsonlines,
    read_txt,
    read_yaml,
    write_json,
    write_jsonlines,
)

for record in iter_json_array_file("corpus.json"):
    ...

write_json("out/records.json", records)  # atomic, ensure_ascii=False
write_jsonlines("out/records.jsonl", rows)  # atomic, takes any iterable
```

Both writers create parent directories and write through a temporary file, so an interrupted run leaves the previous artifact intact rather than a truncated one. Readers default to `utf-8-sig`, which decodes plain UTF-8 unchanged but also strips a byte-order mark.

## llm

```python
from agent_toolkit.llm import complete

answer = await complete("Xin chào", model="gemma-4-31B-it")
```

`complete` returns the answer text. `complete_with_reasoning` returns a `Completion` with `.content` and `.reasoning`; thinking is off by default only if the server says so, and `enable_thinking=True|False` decides it explicitly.

### Configuration

Three resolvers, one installed process-wide. Explicit arguments to `complete` always win over whatever the resolver returns.

```python
from pathlib import Path
from agent_toolkit.llm import (
    LLMConfig,
    DictConfigResolver,
    JsonDirConfigResolver,
    set_config_resolver,
)

# 1. EnvConfigResolver — the default. Reads LLM_MODEL, LLM_API_KEY, LLM_BASE_URL.
#    Nothing to install.

# 2. DictConfigResolver — configs held in memory, keyed by model name.
set_config_resolver(
    DictConfigResolver(
        {
            "gemma-4-31B-it": LLMConfig(
                model="gemma-4-31B-it", api_key=..., base_url=...
            ),
        }
    )
)

# 3. JsonDirConfigResolver — one <model>.json per model in a directory.
#    "GLM-5.1" reads glm-5.1.json; a missing file raises rather than
#    returning a blank config and failing at the request.
set_config_resolver(JsonDirConfigResolver(Path("configs")))
```

`max_tokens`, `temperature`, `max_concurrency`, and `requests_per_minute` are read off the resolved `LLMConfig`, so a resolver can set them per model.

### Structured output

`complete_structured` validates against the schema whichever way the JSON was produced, and returns the reason when it does not conform instead of a half-parsed value.

```python
from agent_toolkit.llm import complete_structured

schema = {"type": "array", "items": {"type": "string", "enum": catalog}}
value, info = await complete_structured(prompt, schema, model="gemma-4-31B-it")

if info.ok:
    use(value)  # validated against `schema`
else:
    abstain(info.error)  # e.g. "$[1]: 'delete_database' is not one of [...]"
```

`mode` picks how the constraint is applied: `"auto"` (default) tries the OpenAI `response_format` and falls back to a prompt instruction, `"grammar"` uses vLLM's `guided_json`, `"prompt"` only asks. A decode-time constraint removes *detectable* invalid output, not invalid output — the module docstring records a measured case where it made the answer worse — so validation runs either way.

### Errors, retry, and traffic

```python
from agent_toolkit.llm import RetryPolicy, TrafficController, set_default_retry_policy
from agent_toolkit.llm.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

set_default_retry_policy(RetryPolicy(max_retries=3, base_delay=1.0))
```

Every provider failure arrives as a subclass of `LLMError`, so one `except LLMError` around a call is enough. Timeouts, 429s, 5xx, and connection drops are retried; authentication, configuration, and other 4xx fail immediately.

`TrafficController` caps in-flight requests and requests per minute. `complete` uses one per model automatically; construct your own only for a pool that needs its own budget.

```python
controller = TrafficController("gemma", max_concurrency=5, requests_per_minute=30)
async with controller:
    ...
```

### Model metadata

```python
from agent_toolkit.llm import (
    count_tokens,
    model_family,
    supports_native_tool_calling,
    supports_reasoning,
)

model_family("Qwen3-8B")  # 'qwen' — an unrecognised name is 'unknown'
count_tokens(messages, "gpt-4o")
```

`count_tokens` estimates with tiktoken and is **rough** — measured drift against a real endpoint runs from −33% to +64% on Vietnamese. Size a request with it; account for what it cost with the `usage` the response reports. It fetches its vocabulary over the network on first use unless `TIKTOKEN_CACHE_DIR` points at a populated cache.

## Logging

`get_logger(__name__)` returns a standard library logger and does nothing else. The library adds no handler and sets no level, so a host that configures nothing sees nothing.

## Not in 0.1

The spec's public surface lists these; they are not implemented, so importing them fails rather than silently doing something else.

| Deferred | Why |
|---|---|
| `llm.stream` | No v0.1 consumer streams. It also needs the `TrafficController` fix its harvested caller depends on — `_wait_for_token()` does not exist, so every streaming call raised `AttributeError`. The fix ships with the feature. |
| `llm.complete_with_tools`, `extract_tool_calls_from_text` | Jurors return JSON text, not tool calls. |
| `json_utils.loads_repair`, `deep_merge`, `json_diff`, `jsonpath_get` | No v0.1 consumer. `iter_json_array` is the one the pipeline needs. |
| Embeddings, Responses API conversion, a provider abstraction | The pipeline uses local static embeddings and one OpenAI-compatible path. |
| Full behavioural parity with `voice-agent-toolkit` 0.2.24, and the `agent-evaluation` migration | Parity exists to make that migration provably transparent. DataForce is greenfield and carries no parity obligation. |
| Registry publishing, 3.11 and 3.14 wheels | The pipeline installs from a path or a git ref on 3.12. |

Also known and unaddressed: `LLMRateLimitError.retry_after` is never populated from a `Retry-After` header, and `complete` discards the response's `usage`.

## Develop

```bash
./setup-dev-uv.sh
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src/agent_toolkit
uv run pytest -q
```

Two tests exercise `iter_json_array` against a real multi-hundred-megabyte JSON array and are skipped unless you point them at one:

```bash
AGENT_TOOLKIT_CORPUS=/path/to/array.json uv run pytest -q -k RealCorpus
# override the expected element count if it is not 21,172:
#   AGENT_TOOLKIT_CORPUS_COUNT=1000
```

`tests/consumer_smoke.py` is not part of that suite. It checks the fifteen symbols the pipeline imports against an *installed wheel*, calling each one once, with the two LLM entry points pointed at a stub HTTP server. The unit tests all replace the transport, so this is the only check that the wheel works with its real resolved dependencies:

```bash
uv build
python3.12 -m venv /tmp/at-full
/tmp/at-full/bin/pip install "dist/agent_toolkit-0.1.0-py3-none-any.whl[llm]"
/tmp/at-full/bin/python tests/consumer_smoke.py
```
