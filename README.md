# agent-toolkit

Shared utilities for agent and dataset pipelines: string, JSON, and file helpers, plus an OpenAI-compatible LLM client with retry and rate limiting.

Spec: [`docs/spec.md`](docs/spec.md). Plan: [`docs/plan.md`](docs/plan.md).

Two guarantees shape the design:

- **The core is light.** `string_utils`, `json_utils`, and `file_utils` depend on `json-repair` and `pyyaml` only. Importing them does not import the OpenAI SDK.
- **The library configures nothing.** No logging handler, no logger level, no environment variable read at import time. The host owns all of that.

## Install

`agent-toolkit` is not published to PyPI or any registry. Install it from this repository by git ref, pinned to a tag.

```bash
pip install "agent-toolkit @ git+https://github.com/giangchicken/agent-toolkit.git@v0.1.0"          # core
pip install "agent-toolkit[llm] @ git+https://github.com/giangchicken/agent-toolkit.git@v0.1.0"     # core + LLM client
```

Python 3.11 or newer. Take the `[llm]` extra unless you only want the string, JSON, and file helpers; importing `agent_toolkit.llm` without it raises an `ImportError` naming the extra rather than failing later with `No module named 'openai'`.

### Depending on it from another project

Pin the tag, not a branch. `main` moves, and a resolver that sees a moving ref will happily install a different library into a rebuilt environment than the one your last run was tested against.

**`pyproject.toml`** — a direct URL dependency, which is what PEP 621 gives you without a registry:

```toml
[project]
dependencies = [
    "agent-toolkit[llm] @ git+https://github.com/giangchicken/agent-toolkit.git@v0.1.0",
]
```

**`requirements.txt`**:

```
agent-toolkit[llm] @ git+https://github.com/giangchicken/agent-toolkit.git@v0.1.0
```

**uv**:

```bash
uv add "agent-toolkit[llm] @ git+https://github.com/giangchicken/agent-toolkit.git@v0.1.0"
```

**Poetry** needs its own spelling, since it does not read PEP 508 direct-URL strings:

```toml
[tool.poetry.dependencies]
agent-toolkit = { git = "https://github.com/giangchicken/agent-toolkit.git", tag = "v0.1.0", extras = ["llm"] }
```

Two things to know before you add it to a build. A git dependency needs `git` on the machine that installs it, which rules out most minimal container images and some CI runners unless you add it. And a private-repo install needs credentials at install time — a deploy key, or `git+ssh://git@github.com/...` with an agent — because pip shells out to `git` and inherits nothing else.

**Local checkout**, for developing the two side by side:

```bash
pip install -e "/path/to/agent-toolkit[llm]"
uv add --editable "/path/to/agent-toolkit[llm]"
```

### Checking the install

`tests/consumer_smoke.py` calls each of the fourteen symbols a consumer imports, against whatever `agent_toolkit` is on the path. It needs no network beyond a localhost socket, because the two LLM entry points talk to a stub `http.server` it starts itself. Run it inside your own environment to confirm the dependency resolved to something that works:

```bash
git clone --depth 1 -b v0.1.0 https://github.com/giangchicken/agent-toolkit.git /tmp/at
python -m pip install "agent-toolkit[llm] @ git+https://github.com/giangchicken/agent-toolkit.git@v0.1.0"
python /tmp/at/tests/consumer_smoke.py
```

It prints one line per symbol and exits non-zero on the first that misbehaves. Note that it is not in the wheel — wheels carry `src/agent_toolkit` only — so it comes from a checkout or from the sdist.

## string_utils

The tables these read — `THINKING_MARKERS`, `SPOKEN_DIGITS`, `SPOKEN_AT`, `SPOKEN_DOT`, `NAME_TITLES`, `OTP_CUES`, `SPACE_UNICODES` — live in `agent_toolkit.lexicon`, a `shape` module that imports nothing, so a host can read or extend a language table without importing the rules over it.

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
# opening tag and a block that max_tokens cut off, which a two-tag regex does not.
split_thinking("Chào hỏi thôi.</think>Xin chào!")
# ('Chào hỏi thôi.</think>', 'Xin chào!')
```

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

Four resolvers, one installed process-wide. Explicit arguments to `complete` always win over whatever the resolver returns.

```python
from pathlib import Path
from agent_toolkit.llm import (
    LLMConfig,
    DictConfigResolver,
    JsonDirConfigResolver,
    YamlConfigResolver,
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

# 4. YamlConfigResolver — every model in one file, with shared defaults.
set_config_resolver(YamlConfigResolver("models.yaml"))
```

#### One YAML file for every model

The resolver to reach for when the question is "what temperature does this model run at" — the answer for all of them is in one place, and a model that names nothing takes the defaults rather than being underspecified. See [`examples/models.yaml`](examples/models.yaml).

```yaml
defaults:
  temperature: 0.3
  top_p: 1.0
  max_tokens: 4096
  timeout: 120.0
  max_concurrency: 10
  requests_per_minute: 600

models:
  gemma-4-31B-it: {}          # every default as-is
  DeepSeek-V4-Flash:
    enable_thinking: false
    max_tokens: 8012
  Qwen3.6-27B:
    temperature: 0
    enable_thinking: false
```

```python
set_config_resolver(YamlConfigResolver("models.yaml"))

answer = await complete(
    "Xin chào",
    model="Qwen3.6-27B",
    api_key=os.environ["FPT_API_KEY"],
    base_url=os.environ["FPT_BASE_URL"],
)
```

Settings resolve in four steps, each beating the one below it: the argument at the call site, the model's own block, the `defaults` block, the `LLMConfig` field default.

**No `api_key` and no `base_url` in the file.** This file says how a model behaves, not where it runs or what authenticates it — both are the caller's to supply. Writing either one here raises rather than being quietly used or quietly ignored: a credential in a committed file is a leak, and an endpoint in one is how a staging run reaches production. Pass them at the call site, or install a `DictConfigResolver` if the host prefers to inject them once.

Unknown keys raise too, at load time rather than at the first call — this is one file naming every model for a whole run, so `temperatur: 0` would otherwise leave every call at 0.7 with nothing saying so. `JsonDirConfigResolver` still ignores unknown keys, since those files predate the toolkit.

`max_tokens`, `temperature`, `top_p`, `timeout`, `enable_thinking`, `max_concurrency`, and `requests_per_minute` are read off the resolved `LLMConfig`, so a resolver can set them per model. `top_p` and `enable_thinking` are sent only when something sets them; `timeout` configures the HTTP client rather than the request body.

### Errors, retry, and traffic

```python
from agent_toolkit.llm import TrafficController
from agent_toolkit.llm.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

await complete(prompt=..., model=..., max_retries=1, retry_delay=0.0)
```

Retry is configured per call, by the three keywords `agent-evaluation`'s `complete()` takes: `max_retries` (default 8), `retry_delay` (5.0 seconds, and the multiplier when backoff is exponential), and `exponential_backoff` (True). They are named parameters, so they configure the retry rather than falling through `**kwargs` into the request body, and each attempt waits `retry_delay * 2 ** (n - 1)` capped at two minutes.

The three defaults are module constants — `DEFAULT_MAX_RETRIES`, `DEFAULT_RETRY_DELAY`, `DEFAULT_EXPONENTIAL_BACKOFF` on `agent_toolkit.llm.factory`, importable from `agent_toolkit.llm`. They are bound into the signatures at import, as in the harvested code, so reassigning one later does not change an unset keyword; pass the keyword at the call site, which always wins.

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

## embed

```python
from agent_toolkit.embed import vectors

rows = await vectors(["câu thứ nhất", "câu thứ hai"], model="bge-m3")
```

One vector per text, in the order the texts were given — paired by the response's `index`, not by arrival order, because a mis-paired vector is not something anything downstream of it can detect. `batch_size=` splits one call into several requests dispatched concurrently under the model's own concurrency budget; left unset, one request carries every text. A provider that answers with fewer rows than it was asked about raises `EmbedProviderError` rather than returning a shifted list.

`agent_toolkit.embed` mirrors `agent_toolkit.llm` module for module — its own `config.py`, `exceptions.py`, `error_mapping.py`, `traffic_control.py`, `executors.py`, `factory.py` — and three consequences follow from that:

- **Its own resolver.** `agent_toolkit.embed.set_config_resolver()` is a different process-wide slot from the chat route's. A host that configures both installs one on each; installing a resolver on `agent_toolkit.llm` leaves `vectors()` on `EMBED_MODEL` / `EMBED_API_KEY` / `EMBED_BASE_URL`.
- **Its own traffic budget.** `agent_toolkit.embed.get_traffic_controller("m")` and `agent_toolkit.llm.get_traffic_controller("m")` are two controllers, so a model called from both routes is throttled once per route rather than once in total.
- **Its own error taxonomy.** `vectors` raises `EmbedError` subclasses, never `LLMError` ones. A host that wrapped both routes in one `except LLMError` needs a second clause after this.

`EmbedConfig` carries only what an embeddings call needs — `model`, `api_key`, `base_url`, `binding`, `extra_headers`, `timeout`, `max_concurrency`, `requests_per_minute`. It has no `max_tokens`, `temperature`, `top_p`, `reasoning_effort`, `enable_thinking` or `api_version`: those are the body of the chat route, and a field here would be a number a resolver could set and this route would never send. The four resolvers and the precedence rule are the chat route's, unchanged, and a YAML file naming a chat-only setting raises rather than ignoring it.

```python
from agent_toolkit.embed.exceptions import EmbedError, EmbedInputTooLargeError
```

`agent_toolkit.embed.exceptions` is ten classes under `EmbedError`, shaped like the chat route's twelve and named for this route: `EmbedConfigError`, `EmbedProviderError`, `EmbedAPIError` and its four status-carrying children, `EmbedQuotaExceededError`, and `EmbedInputTooLargeError`. There is no parse error and no circuit breaker — nothing here reads a model's prose.

Two things follow from the taxonomy being this route's own rather than the chat route's. `EmbedInputTooLargeError` means *one text over the model's token limit*, which is a length the caller fixes by chunking, so it is **not retried** — where the chat route's `ProviderContextWindowError` sets no status code, falls through to the catch-all, and spends the whole backoff schedule on a certainty. And the retry defaults are `agent_toolkit.embed.factory`'s own constants — `DEFAULT_MAX_RETRIES` / `DEFAULT_RETRY_DELAY` / `DEFAULT_EXPONENTIAL_BACKOFF`, importable from `agent_toolkit.embed`, same values and same three keywords per call — so tuning one route's retry no longer moves the other's.

## Logging

`get_logger(__name__)` returns a standard library logger and does nothing else. The library adds no handler and sets no level, so a host that configures nothing sees nothing. Every module logs through it: the config resolver's answer, each call's model and endpoint, and failures — all on the `agent_toolkit` logger tree, at `DEBUG` except warnings and errors.

A host that wants to *see* those records configures the `agent_toolkit` logger itself, or calls `configure_logging()` for the console-and-file setup `agent-evaluation` uses:

```python
import agent_toolkit

agent_toolkit.configure_logging("DEBUG", log_dir="logs")
# [llm.factory] DEBUG: LLM call: model=glm-5.1 binding=openai base_url=... max_tokens=4096 ...
# [llm.factory] DEBUG: LLM call ok in 1.83s: model=glm-5.1 content=412 chars reasoning=0 chars
# [embed.factory] DEBUG: embeddings call ok in 0.21s: model=bge-m3 vectors=2 dimensions=1024
```

It installs a colored, level-tagged stderr handler and — when `log_dir` is given — a daily `agent_toolkit_YYYYMMDD.log` beneath it, both on the `agent_toolkit` logger. Nothing happens at import; the handlers appear only when the host calls it, it is idempotent (a second call replaces its own handlers and no one else's), and `propagate=False` by default keeps the records out of a host's root handlers. `console=False` gives file-only output, `colors=` forces the ANSI codes on or off, and `file_level=` sets the file's threshold independently of the console's.

## Not in 0.1

The spec's public surface lists these; they are not implemented, so importing them fails rather than silently doing something else.

| Deferred | Why |
|---|---|
| `llm.stream` | No v0.1 consumer streams. It also needs the `TrafficController` fix its harvested caller depends on — `_wait_for_token()` does not exist, so every streaming call raised `AttributeError`. The fix ships with the feature. |
| `llm.complete_with_tools`, `extract_tool_calls_from_text` | Jurors return JSON text, not tool calls. |
| `json_utils.loads_repair`, `deep_merge`, `json_diff`, `jsonpath_get` | No v0.1 consumer. `iter_json_array` is the one the pipeline needs. |
| Responses API conversion, a provider abstraction | One OpenAI-compatible path is what the pipeline calls. |
| Full behavioural parity with `voice-agent-toolkit` 0.2.24, and the `agent-evaluation` migration | Parity exists to make that migration provably transparent. DataForce is greenfield and carries no parity obligation. |
| Registry publishing, 3.11 and 3.14 wheels | The pipeline installs from a path or a git ref on 3.12. |

Also known and unaddressed: `LLMRateLimitError.retry_after` is never populated from a `Retry-After` header, and `complete` discards the response's `usage` — so the call logs report elapsed time and response size, not tokens.

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

`tests/consumer_smoke.py` is not part of that suite. It checks the fourteen symbols the pipeline imports against an *installed wheel*, calling each one once, with the two LLM entry points pointed at a stub HTTP server. The unit tests all replace the transport, so this is the only check that the wheel works with its real resolved dependencies:

```bash
uv build
python3.12 -m venv /tmp/at-full
/tmp/at-full/bin/pip install "dist/agent_toolkit-0.1.0-py3-none-any.whl[llm]"
/tmp/at-full/bin/python tests/consumer_smoke.py
```
