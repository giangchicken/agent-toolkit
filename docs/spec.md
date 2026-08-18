# agent-toolkit — Shared Python Library

## What

`agent-toolkit` is a standalone, installable Python library holding the utilities that agent projects keep rewriting: `{{placeholder}}` slot filling, tolerant JSON extraction from LLM output, file/JSON I/O including streaming readers for 100MB+ files, and a provider-agnostic async LLM client with retry, error mapping, and traffic control. It lives in its own repository, is published to the internal package registry, and is consumed by version pin — first by DataForce, then by `agent-evaluation`, replacing that project's `voice-agent-toolkit` dependency and its partially de-vendored local copies.

## Context

Two things in `agent-evaluation-dev` motivate this library, and both are visible in the source.

**There is already a toolkit, and the project is halfway off it.** `pyproject.toml:9` pins `voice-agent-toolkit` from the internal GitLab by git tag (`@refs/tags/0.2.24`). Sixteen call sites import from it:

| Symbol | Call sites |
|---|---|
| `voice_agent_toolkit.string_utils.slot_filling` | 8 |
| `voice_agent_toolkit.string_utils.extract_json_from_text` | 8 |
| `voice_agent_toolkit.file_utils.read_json` | `src/components/llm/config.py:16` |
| `voice_agent_toolkit.file_utils.read_txt` | `src/dependencies.py:18` |
| `voice_agent_toolkit.agent.tool_utils.extract_tool_calls_from_text` | `src/agents/evaluations/llm/function_calling/parser.py:14` |

Meanwhile `src/utils/string_utils.py:1-17` describes itself as *"Toolkit-free replacements for the two `voice_agent_toolkit.string_utils` helpers"*, and states the extraction contract outright: *"Behavior is copied from the toolkit … so a later swap of the toolkit callers onto this module is transparent."* The seam this spec needs already exists — the project has been drifting toward it, one file at a time, without a destination. This library is that destination.

**The LLM component is already library-shaped.** `src/components/llm/` (1,354 lines across 8 files) is self-contained and deliberately toolkit-free — `utils.py:17` notes the reasoning-model patterns are *"Kept local so this component has no external (toolkit) dependency."* It provides `complete` / `stream` / `complete_with_tools` over any OpenAI-compatible endpoint, a `LLMError` hierarchy, provider error mapping, tenacity retry with retriable-error classification, and a `TrafficController` combining an `asyncio.Semaphore` concurrency gate with a token-bucket rate throttle. That is a real, non-trivial LLM client, and DataForce needs exactly it.

What blocks a straight copy is host coupling, in three places, plus four defects worth fixing on the way out rather than carrying forward. Both are enumerated under Design.

## Requirements

1. The distribution is named `agent-toolkit`; the import package is `agent_toolkit`. It builds a wheel and publishes to the internal package registry under SemVer git tags, matching how `voice-agent-toolkit` is consumed today.
2. `requires-python = ">=3.11,<4"`. This is a hard constraint, not a preference: `agent-evaluation` pins `requires-python = "==3.12.3"` and DataForce targets 3.14, and the library must install into both.
3. Core modules (`string_utils`, `json_utils`, `file_utils`) depend only on `json-repair` and `pyyaml`. The LLM client is an optional extra, `agent-toolkit[llm]`, pulling `openai`, `tenacity`, and `aiohttp`. Importing `agent_toolkit.string_utils` must not import the OpenAI SDK.
4. The library configures no logging handlers and reads no environment variable at import time. It calls `logging.getLogger(__name__)` and nothing more.
5. The library knows nothing about any host application's directory layout. No module may reference a `configs/` or `prompts/` path.
6. These symbols keep the exact names and signatures the current call sites use, so migration is an import-line change and nothing else: `string_utils.slot_filling`, `string_utils.extract_json_from_text`, `file_utils.read_json`, `file_utils.read_txt`, `llm.complete`, `llm.stream`, `llm.complete_with_tools`.
7. `json_utils.iter_json_array` streams a top-level JSON array from a file, yielding one element at a time with memory proportional to the largest element, not the file. It must handle a 127MB array of 21k objects without loading it whole.
8. `file_utils.write_json` writes atomically — temp file in the destination directory, then `os.replace` — so a crash mid-write cannot truncate an existing file.
9. The LLM client accepts configuration by explicit argument, by injected resolver, or from environment variables, in that precedence order.
10. The package ships `py.typed` and passes `mypy --strict` on its public surface.
11. Every public function has behavior pinned by a test derived from the current toolkit's observed output, so the swap in `agent-evaluation` is provably transparent rather than assumed to be.

## Design

### Layout

```
agent-toolkit/
├── src/agent_toolkit/
│   ├── __init__.py           re-exports the stable surface; __version__
│   ├── string_utils.py       slot_filling, extract_json_from_text, clean_thinking_tags
│   ├── json_utils.py         loads_repair, iter_json_array, deep_merge, json_diff, jsonpath_get
│   ├── file_utils.py         read_json, read_txt, read_yaml, write_json, iter_json_array_file
│   ├── logging.py            get_logger — thin logging.getLogger wrapper, no handler setup
│   ├── errors.py             ToolkitError base
│   ├── llm/
│   │   ├── __init__.py       complete, stream, complete_with_tools
│   │   ├── config.py         LLMConfig, ConfigResolver, EnvConfigResolver, DictConfigResolver
│   │   ├── executors.py      sdk_complete, sdk_stream, sdk_complete_with_tools
│   │   ├── exceptions.py     LLMError hierarchy
│   │   ├── error_mapping.py  map_error
│   │   ├── retry.py          RetryPolicy, is_retriable
│   │   └── traffic_control.py TrafficController
│   └── py.typed
├── tests/
└── pyproject.toml
```

`string_utils.py` and the whole `llm/` tree are lifted from `agent-evaluation-dev` — `src/utils/string_utils.py` and `src/components/llm/` respectively — with the changes below. Everything else is new.

### Breaking the host coupling

Three imports make the current LLM component unliftable. Each has one correct fix.

**1. `config.py:22` hardcodes the host's config directory.**

```python
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"
```

`get_llm_config("GLM-5.1")` lowercases the model name into `glm-5.1.json` and reads it from that directory. A library cannot reach four parents up and expect a `configs/` folder. Replace with a resolver protocol the host supplies:

```python
class ConfigResolver(Protocol):
    def resolve(self, model: str | None) -> LLMConfig: ...

class EnvConfigResolver:      # default — LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, ...
class DictConfigResolver:     # in-memory mapping, for tests and DataForce
class JsonDirConfigResolver:  # <dir>/<model>.json — preserves agent-evaluation's behavior
```

`agent-evaluation` keeps working by installing `JsonDirConfigResolver(Path("configs"))` once at startup. DataForce uses `DictConfigResolver` fed from its own project settings.

**2. `factory.py:35` imports the host's settings object.**

```python
from src.dependencies import settings
DEFAULT_MAX_RETRIES = settings.retry.max_retries
```

Replace with an explicit `RetryPolicy` dataclass — `max_retries=8`, `base_delay=5.0`, `exponential_backoff=True`, matching `LLMRetryConfig`'s current defaults — passed per call or set once via `set_default_retry_policy()`.

**3. `executors.py:12` and `traffic_control.py:22` import the host's logger.**

`from src.logging import get_logger` becomes `from agent_toolkit.logging import get_logger`, which is `logging.getLogger(name)` and nothing else. The host owns handler configuration.

### Defects fixed during extraction

These are in the code being lifted. They are fixed in the library, not carried over.

**`stream()` with a traffic controller always raises `AttributeError`.** `factory.py:378` calls `await _traffic_ctrl._wait_for_token()`. `TrafficController` defines no such method — its methods are `_acquire_credit`, `__aenter__`, `__aexit__`, and two properties. Because `get_llm_config` unconditionally constructs a `TrafficController` (`config.py:114`), `_traffic_ctrl` is never `None`, so the guarded branch is the only branch and **every streaming call fails**. The fix is to use the context manager the class already provides, as `complete()` correctly does:

```python
async with _traffic_ctrl:
    async for chunk in sdk_stream(...):
        yield chunk
```

Note this changes hold semantics: the concurrency slot is held for the whole stream rather than released early. That is the correct behavior for a streaming call — the request *is* in flight for its whole duration — and it is what the current code was evidently reaching for.

**`_acquire_credit` recurses instead of looping.** `traffic_control.py:100` calls itself after sleeping. Under sustained throttling with many waiters, each retry adds a frame. Rewrite as a `while True` loop; the logic is unchanged.

**The cached `TrafficController` outlives its event loop.** `get_llm_config` is `@lru_cache(maxsize=None)` and builds a `TrafficController` holding an `asyncio.Semaphore`. The cache is process-global, so a second `asyncio.run()` in the same process — routine in tests and CLI scripts — reuses a semaphore whose waiters belong to a closed loop. Key the controller by `(model, id(running_loop))` and build it lazily on first use inside a running loop.

**`active_requests` reads a CPython private.** `traffic_control.py:110` returns `self.max_concurrency - self._semaphore._value`. Track an explicit `_in_flight` counter incremented in `__aenter__` and decremented in `__aexit__`.

### Streaming JSON reader

DataForce imports `fc_train_final.json` — 127MB, a top-level array of 21,172 objects. `json.load` on it costs roughly 1.5GB of resident memory. `json_utils.iter_json_array` reads a bounded buffer and repeatedly applies `json.JSONDecoder().raw_decode` at the next value boundary, yielding elements and compacting the buffer as it goes:

```python
def iter_json_array(fp: IO[str], *, buffer_size: int = 1 << 20) -> Iterator[Any]:
    """Yield elements of a top-level JSON array one at a time.

    Memory is bounded by buffer_size plus the largest single element.
    Raises ToolkitError if the document's top level is not an array.
    """
```

`file_utils.iter_json_array_file(path)` wraps it with encoding handling. The pure-stdlib approach is deliberate: it keeps the core extra-free, and adding `ijson` would put a C extension in the dependency path of every consumer for one function.

### Public surface

```python
from agent_toolkit.string_utils import slot_filling, extract_json_from_text, clean_thinking_tags
from agent_toolkit.json_utils  import loads_repair, iter_json_array, deep_merge, json_diff, jsonpath_get
from agent_toolkit.file_utils  import read_json, read_txt, read_yaml, write_json, iter_json_array_file
from agent_toolkit.llm import complete, stream, complete_with_tools, LLMConfig, RetryPolicy, TrafficController
from agent_toolkit.llm.exceptions import LLMError, LLMAPIError, LLMRateLimitError, LLMTimeoutError
```

`slot_filling` and `extract_json_from_text` keep their current signatures exactly, including `extract_json_from_text(text, extract_all=False)` returning a parsed `dict`/`list`/`None` rather than a string.

### Migration of `agent-evaluation`

One commit, mechanical: add `agent-toolkit[llm]`, drop `voice-agent-toolkit`, rewrite 16 import lines, install `JsonDirConfigResolver(Path("configs"))` and the `RetryPolicy` at startup, delete `src/components/llm/` and `src/utils/string_utils.py`. `extract_tool_calls_from_text` is the one symbol with no v1 home (see Out of Scope) — `src/agents/evaluations/llm/function_calling/parser.py` keeps its `voice_agent_toolkit` import until v1.1, so both toolkits are installed during that window. That overlap is the price of not rushing an unread function into the library.

## Decisions

**Separate repository, not a monorepo package.** *Alternatives:* `packages/agent-toolkit` inside DataForce. *Why:* the library's second consumer, `agent-evaluation`, is a different team's repo; making it depend on DataForce's repo inverts ownership. A separate repo with registry publishing matches how `voice-agent-toolkit` is already consumed, so the install story needs no new infrastructure. *Reversible:* yes, but the import path would churn for both consumers.

**Extract fresh from `agent-evaluation`, not fork `voice-agent-toolkit`.** *Alternatives:* fork 0.2.24, strip voice-specific code, rename. *Why:* `agent-evaluation` was already walking away from that toolkit one module at a time, and the local LLM component was written specifically to avoid depending on it. Forking inherits whatever drove that, unread. Extracting takes code this spec has read line by line. *Reversible:* no — this is which codebase we start from.

**LLM client behind an optional extra.** *Alternatives:* one flat dependency set. *Why:* DataForce's API container parses JSON on every import path and calls an LLM on none of them; it should not pull the OpenAI SDK, `tenacity`, and `aiohttp` to use `slot_filling`. *Reversible:* yes, additive.

**Config by injected resolver, defaulting to environment.** *Alternatives:* keep the `configs/<model>.json` convention as library behavior. *Why:* a library that reads a directory relative to its own `__file__` is not a library. The resolver preserves the existing convention for `agent-evaluation` while letting DataForce store LLM settings in Postgres. *Reversible:* yes.

**Pure-stdlib streaming JSON.** *Alternatives:* depend on `ijson`. *Why:* one function does not justify a C extension in every consumer's dependency tree; `raw_decode` over a sliding buffer is ~40 lines and testable. *Reversible:* yes — `ijson` can back the same signature later if profiling demands it.

**Assumption:** the internal GitLab package registry is the publish target, and this repo sits alongside the existing `voice-agent-toolkit` project. Confirm the group and the CI credentials before the first tag.

**Assumption:** `voice_agent_toolkit.agent.tool_utils.extract_tool_calls_from_text` is the only symbol from the old toolkit not reproduced in v1. If other internal projects depend on more of it, this library's surface needs a second pass before they can migrate.

## Versions

| Component | Pinned | Source |
|---|---|---|
| Python | `>=3.11,<4` | Constrained by `agent-evaluation` (`==3.12.3`) and DataForce (3.14.7) |
| json-repair | `>=0.59.10` | Matches `agent-evaluation`'s current pin |
| pyyaml | `>=6.0.3` | Matches current pin |
| openai *(extra)* | `>=2.0` | Current SDK line; verify at scaffold |
| tenacity *(extra)* | `>=9.1.4` | Matches current pin |
| aiohttp *(extra)* | latest stable | Used only by retry classification |
| Build backend | `hatchling` | Standard for `src/` layout |
| Lint/format | `ruff`, `mypy --strict` | `agent-evaluation` already runs pre-commit |

## Invariants

1. **No host coupling.** No module imports `src.*`, references a path outside the package, or reads an env var at import time. *Check:* an `import-linter` contract in CI plus a grep test asserting zero matches for `from src\.` and `Path(__file__).*parent.*configs`.
2. **Core stays light.** `import agent_toolkit.string_utils` does not import `openai`. *Check:* a test that imports the core modules in a subprocess and asserts `"openai" not in sys.modules`.
3. **Behavioral parity.** `slot_filling` and `extract_json_from_text` produce output identical to `voice-agent-toolkit` 0.2.24 for every recorded case. *Check:* a golden corpus captured from the installed toolkit before extraction, replayed against the new implementation. This is the test `src/utils/string_utils.py` promised transparency without.
4. **Streaming stays bounded.** `iter_json_array` over `fc_train_final.json` peaks under 100MB RSS. *Check:* a `tracemalloc` assertion in the test suite against a generated fixture of comparable shape.
5. **No silent logging config.** Importing any module adds no handler to the root logger. *Check:* assert `logging.getLogger().handlers` is unchanged across import.

## Error Behavior

The `LLMError` hierarchy is preserved exactly as it exists today — `LLMError` → `LLMConfigError`, `LLMProviderError`, `LLMCircuitBreakerError`, `LLMAPIError` → `LLMTimeoutError` (408), `LLMRateLimitError` (429, carries `retry_after`), `LLMAuthenticationError` (401), `LLMModelNotFoundError` (404) — so existing `except` clauses in `agent-evaluation` keep working.

Retry classification also carries over unchanged: retry on timeouts, 429, 5xx, and connection errors; never on `CancelledError`, `KeyboardInterrupt`, `GeneratorExit`, auth errors, config errors, or 4xx other than 429. `LLMRateLimitError.retry_after` raises the floor on the computed backoff.

Two behaviors are pinned that are currently implicit:

- **A stream that has already yielded never retries.** Retrying mid-stream would emit duplicate content to the caller. `factory.py:418` gets this right via `has_yielded`; the library states it as contract and tests it.
- **Core utilities never raise.** `slot_filling`, `extract_json_from_text`, and the readers log at debug and return a fallback (input text, `None`, empty dict) rather than propagating. This matches current behavior and the call sites that depend on it. `iter_json_array` is the deliberate exception — a malformed 127MB import must fail loudly, not yield a truncated dataset.

## Testing Strategy

- **Parity corpus (do this first).** Before writing any implementation, capture `slot_filling` and `extract_json_from_text` outputs from the installed `voice-agent-toolkit` 0.2.24 across a corpus drawn from `agent-evaluation`'s real prompt files and recorded LLM outputs — fenced blocks, bare objects, bare arrays, nested candidates, malformed JSON that `json_repair` recovers, nested `{{placeholder}}` chains. Commit it as a fixture. Everything after is measured against it.
- **Unit:** each utility, including the edge cases the source comments call out — `extract_response_content` returning `""` rather than a repr for complex SDK objects, `clean_thinking_tags` on unclosed tags, `slot_filling` leaving unknown placeholders untouched and terminating on self-referential input.
- **Streaming:** `iter_json_array` against a generated 100MB+ fixture; correctness against the same data parsed by `json.load`; the memory assertion from invariant 4; a malformed-array case asserting a loud failure.
- **LLM client:** `respx`-mocked OpenAI endpoints covering retry-then-succeed, retry-exhausted, non-retriable-fails-fast, and rate-limit honoring `retry_after`. A regression test for the `stream()` + `TrafficController` path that currently raises `AttributeError` — that test fails against today's code and passes against the fix.
- **TrafficController:** token-bucket refill accuracy under a fake clock; concurrency never exceeding `max_concurrency`; sustained throttling adding no stack depth; the same cached controller reused across two successive `asyncio.run()` calls.
- **Packaging:** build the wheel and install it into clean 3.11, 3.12, and 3.14 virtualenvs; import core without the extra; import `agent_toolkit.llm` with it and assert a clear `ImportError` without it.
- **Migration proof:** run `agent-evaluation`'s existing test suite against a local install of `agent-toolkit` with the imports rewritten. That suite passing is the definition of a transparent swap.

## Out of Scope

- `extract_tool_calls_from_text` and the `ToolDefinition` / `ToolParameter` / `ToolResult` protocol from `src/core/tool_protocol.py`. Deferred to v1.1 — the tool protocol is worth sharing, but `extract_tool_calls_from_text` lives in `voice-agent-toolkit` and this spec has not read it; specifying a port of unread code would be guessing.
- `PromptManager` and `load_prompt` from `src/dependencies.py`. Their resolution order (`<agent>_<lang>.txt` → `<agent>.txt` → `<lang>.txt` → `default.txt`) is a host convention, not a library one.
- Sync wrappers. The LLM client is async-only, as it is today.
- Anthropic, Gemini, or Bedrock bindings beyond the OpenAI-compatible path. The `binding` parameter is carried through but only `"openai"` is implemented, matching current behavior.
- Streaming tool calls, structured outputs, prompt caching, and token counting.
- Retiring `voice-agent-toolkit`. Other internal projects consume it; this library only removes `agent-evaluation`'s dependency on most of it.
