# agent-toolkit — Implementation Plan (v0.1-minimal)

Source spec: [`spec.md`](./spec.md). The consumer that motivated it is the SFT dataset pipeline, specified in the DataForce repository.

This plan builds the **smallest library that unblocks the pipeline's S0 smoke run**, not the full library the spec describes. Everything cut is listed in [v0.2](#deferred-to-v02) with the reason.

## Local paths

The two source checkouts are not committed here. Set these before starting; every file reference below is relative to one of them.

| Var | Points at |
|---|---|
| `$AE` | the `agent-evaluation` checkout — supplies the **reliability** layer (`src/components/llm/`, `src/utils/string_utils.py`) |
| `$VAT` | the `voice-agent-toolkit` checkout — supplies the **capability** layer (`voice_agent_toolkit/`) |
| `$CORPUS` | the Tool-Decision corpus `fc_train_final.json` (126 MiB, 21,172 records) |
| `$TK` | the new `agent-toolkit` repo being built |

## Spec deltas this plan assumes

The spec was written before either source was read end to end. Five decisions changed; the spec needs a matching amendment, and this plan proceeds on the corrected versions.

**D1 — Scope is v0.1-minimal.** In: `string_utils`, `json_utils`, `file_utils`, `logging`, `errors`, and `llm.complete` with config/retry/traffic/errors. Out: `stream`, `complete_with_tools`, the `agent-evaluation` migration, registry publishing, and the 3.11/3.14 wheel matrix.

**D2 — Two-source harvest, copied and never depended on.** The spec's decision *"extract fresh from `agent-evaluation`, not fork `voice-agent-toolkit`"* was made without reading the toolkit. The two clients are complementary, not redundant: `$AE` has retry, a 12-class error taxonomy, and `TrafficController` but no structured output, embeddings, token counting, or model-family detection; `$VAT/llm/openai_client.py` has all of the latter and **zero** retry, semaphore, or backoff. Take reliability from `$AE` and capability from `$VAT`. Copy the code — `$VAT/pyproject.toml` pins `requires-python = "==3.12.3"`, which the pipeline's 3.12.14 does not satisfy, so co-installation is impossible.

**D3 — Structured output and model metadata move into scope.** The spec lists structured outputs as out of scope. The pipeline's jury needs a forced JSON-array response (requirement 18) and model-family detection (requirements 19–20). Both already exist in `$VAT` and are cheaper to harvest than to re-derive.

**D4 — Vietnamese digit/PII classification does *not* enter this library.** `$VAT/text_normalization/digit_processing/` is valuable and gets harvested, but into the **pipeline**, not here. It has one consumer, and growing a shared library for one caller is the failure mode the spec's own key-pool decision names.

**D5 — The parity corpus shrinks.** The spec's Testing Strategy opens with "capture a parity corpus from the installed toolkit before implementing." `$VAT/tests/test_string_utils.py` (12 KB) already exists — harvest it instead. Full behavioural parity gates **v0.2** (the `agent-evaluation` migration), not v0.1, because DataForce is greenfield and carries no parity obligation.

---

## Phase 1 — The package exists and enforces its own constraints

**Goal:** `agent_toolkit` installs, and the two structural guarantees every later module depends on are provable before any of them is written.

### T1 — Scaffold the repository

**Goal.** A `src/`-layout Python package that builds a wheel, installs editable, and lints/type-checks clean with no modules in it yet.

**Context.** `$VAT` uses a flat package at repo root with hatchling, uv, ruff (E/F/I only, `E501` ignored), and pytest — no mypy, no `py.typed`. Keep the hatchling/uv/ruff choices; add `src/` layout, `py.typed`, and mypy, per spec requirements 1 and 10. `$VAT/setup-dev-uv.sh` and `.pre-commit-config.yaml` are worth copying as-is.

**Relevant files.** `$VAT/pyproject.toml`, `$VAT/setup-dev-uv.sh`, `$VAT/.pre-commit-config.yaml`; spec §Layout.

**Proposed approach.** `pyproject.toml` with `requires-python = ">=3.11,<4"`, hatchling, `[project.optional-dependencies] llm = ["openai>=2.0", "tenacity>=9.1.4", "aiohttp"]`, and core dependencies limited to `json-repair>=0.59.10` and `pyyaml>=6.0.3`. Create the module tree from the spec's Layout section as empty files with docstrings. Add ruff + mypy config and a GitHub Actions workflow running lint, mypy, and pytest on 3.12.

**Acceptance criteria.**
- `uv sync` and an editable install both succeed on Python 3.12.
- Core dependency list is exactly `json-repair` and `pyyaml`; `openai`, `tenacity`, and `aiohttp` appear only under the `llm` extra.
- `ruff check`, `ruff format --check`, and `mypy --strict src/agent_toolkit` all pass on the empty tree.
- The built wheel contains `py.typed`.

**Source reference.** Spec requirements 1, 3, 10; §Layout.

**Verify.**
```
uv sync && uv run ruff check . && uv run mypy --strict src/agent_toolkit && uv run pytest -q
uv build && python -m zipfile -l dist/*.whl | grep py.typed
```

**Out of scope.** Registry publishing and the 3.11/3.14 wheel matrix (v0.2). CI builds 3.12 only.

---

### T2 — `logging` and `errors`, plus the two import-hygiene tests

**Goal.** The library's logging and error base exist, and two tests fail loudly if any future module breaks import hygiene.

**Context.** This task exists because `$VAT` gets both wrong, and the tests are the only thing that keeps us from repeating it. `$VAT/voice_agent_toolkit/__init__.py` attaches a `StreamHandler` to a package-named logger **and** reads `DEBUG_TOOLKIT` from the environment **and** calls `setLevel`, all at import time — so any host importing it inherits stderr output and a logger level it did not choose. Separately, `$VAT` declares 17 hard dependencies with no extras, so `import voice_agent_toolkit.string_utils` pulls pandas, boto3, elasticsearch, kafka, redis, sqlalchemy, and tiktoken.

**Relevant files.** `$VAT/voice_agent_toolkit/__init__.py` (the violation); `$AE/src/components/llm/exceptions.py` (the hierarchy harvested in T6); spec requirements 4, 5 and invariants 2, 5.

**Proposed approach.** `logging.py` exposes `get_logger(name)` returning `logging.getLogger(name)` and nothing else — no handler, no level, no formatter. `errors.py` defines `ToolkitError(Exception)`. `__init__.py` re-exports `__version__` and the stable surface only; it performs no configuration and reads no environment variable.

**Acceptance criteria.**
- Importing any `agent_toolkit` module adds zero handlers to the root logger and to the `agent_toolkit` logger, and changes no logger level.
- No module reads an environment variable at import time.
- A subprocess that imports `agent_toolkit.string_utils` has no `openai` in `sys.modules`.
- A grep-based test finds zero matches for `from src\.` and for a `Path(__file__)`-relative `configs` path anywhere in the package.

**Source reference.** Spec requirements 4, 5; invariants 1, 2, 5.

**Verify.**
```
uv run pytest tests/test_import_hygiene.py -q
uv run python -c "
import logging, sys
before = list(logging.getLogger().handlers)
import agent_toolkit, agent_toolkit.string_utils
assert list(logging.getLogger().handlers) == before, 'handler added at import'
assert 'openai' not in sys.modules, 'core import pulled openai'
print('ok')"
```

**Out of scope.** The `LLMError` hierarchy (T6). This task ships only `ToolkitError`.

---

## Phase 2 — Core utilities work on the real corpus

**Goal.** The pipeline's ingest, validate, and artifact I/O stages could be written against this library. No LLM involved.

### T3 — `string_utils`

**Goal.** `slot_filling`, `extract_json_from_text`, `clean_thinking_tags`, `normalize_text`, and `compute_hash`, with the existing toolkit test suite as the regression gate.

**Context.** `$AE/src/utils/string_utils.py` (171 lines) is a de-vendored copy of the two toolkit helpers and states the transparency goal in its own docstring: *"Behavior is copied from the toolkit … so a later swap of the toolkit callers onto this module is transparent."* It has no test proving it. `$VAT/voice_agent_toolkit/string_utils.py` is the original (`slot_filling` at :151, `extract_json_from_text` at :200) and `$VAT/tests/test_string_utils.py` is the test suite that was missing — harvest it. `normalize_text(text, remove_tone_marks=)` at `$VAT/string_utils.py:27` and `compute_hash` at `:16` are added because the pipeline needs Vietnamese diacritic normalization for dedup and a stable `rid` hash.

**Relevant files.** `$AE/src/utils/string_utils.py`; `$VAT/voice_agent_toolkit/string_utils.py:16,27,151,200`; `$VAT/tests/test_string_utils.py`; `$AE/src/components/llm/utils.py:43` (`clean_thinking_tags`).

**Proposed approach.** Copy `$AE`'s implementations of `slot_filling` and `extract_json_from_text` verbatim — they are the read, reviewed versions — swapping `from src.logging.logger import get_logger` for `agent_toolkit.logging.get_logger`. Copy `normalize_text` and `compute_hash` from `$VAT`. Copy `clean_thinking_tags` from `$AE/src/components/llm/utils.py`. Port `$VAT/tests/test_string_utils.py`, keeping every case that covers a v0.1 symbol and deleting the rest.

**Acceptance criteria.**
- Signatures match the spec exactly, including `extract_json_from_text(text, extract_all=False)` returning a parsed `dict`/`list`/`None` — never a string.
- Every ported test case passes unmodified against the new implementation.
- `slot_filling` resolves nested `{{placeholder}}` chains, leaves unknown placeholders untouched, and terminates on self-referential input.
- `extract_json_from_text` recovers JSON from: a fenced ```json block, a bare object, a bare array, a prose-wrapped object, and a `json_repair`-recoverable malformed object; returns `None` on garbage.
- Both functions log at debug and return a fallback rather than raising, on any input.

**Source reference.** Spec requirements 6, 11; invariant 3 (partial — full parity gates v0.2).

**Verify.**
```
uv run pytest tests/test_string_utils.py -q
uv run python -c "
from agent_toolkit.string_utils import slot_filling, extract_json_from_text
assert slot_filling('{{a}}', {'a':'{{b}}','b':'x'}) == 'x'
assert extract_json_from_text('noise \`\`\`json\n{\"k\":1}\n\`\`\` more') == {'k':1}
assert extract_json_from_text('no json here') is None
print('ok')"
```

**Out of scope.** `extract_xml_from_text`, the number-rendering suite (`split_number_groups`, `render_separated_number`, `is_clock_time`), and `extract_integer_numbers_from_text` — the last three go to the pipeline with the digit classifier, per D4.

---

### T4 — `json_utils.iter_json_array`

**Goal.** Stream a top-level JSON array with memory bounded by the buffer plus the largest element, proven on the 126 MiB corpus.

**Context.** This is the one module with no source to harvest — `$VAT/file_utils.py` has `read_json`, `read_jsonlines`, and friends but no streaming reader. `json.load` on `$CORPUS` costs roughly 1.5 GB resident. The spec chose stdlib `raw_decode` over a sliding buffer rather than depending on `ijson`, to keep the core extra-free. Note the failure mode found while prototyping this: after a successful `raw_decode`, the buffer must be stripped of whitespace **after** the separating comma, not just before it — `raw_decode` does not skip leading whitespace and raises `Expecting value` at position 0, which silently truncates the iteration to a few dozen records instead of erroring.

**Relevant files.** Spec §Streaming JSON reader; `$CORPUS`.

**Proposed approach.** `iter_json_array(fp, *, buffer_size=1<<20)` scans to the opening `[`, then loops: left-strip whitespace, consume any separating commas and the whitespace after them, return on `]`, refill the buffer on an empty buffer or a `ValueError` from `raw_decode`, otherwise yield the decoded element and advance past it. Raise `ToolkitError` when the top-level value is not an array or when the buffer cannot be extended and the remainder does not parse.

**Acceptance criteria.**
- Yields exactly 21,172 elements from `$CORPUS`, and every element equals the corresponding element from `json.load` on a smaller fixture.
- Peak RSS stays under 100 MB while iterating a fixture of `$CORPUS`'s shape.
- A truncated or malformed array raises `ToolkitError` — it never yields a short iteration silently. This is the deliberate exception to "core utilities never raise."
- Handles elements larger than `buffer_size`, and an array containing exactly zero elements.
- `file_utils.iter_json_array_file(path)` wraps it with encoding handling.

**Source reference.** Spec requirement 7; invariant 4; error behavior ("a malformed 127MB import must fail loudly, not yield a truncated dataset").

**Verify.**
```
uv run pytest tests/test_json_utils.py -q
uv run python -c "
import tracemalloc
from agent_toolkit.file_utils import iter_json_array_file
tracemalloc.start()
n = sum(1 for _ in iter_json_array_file('$CORPUS'))
peak = tracemalloc.get_traced_memory()[1] / 1e6
print(n, f'{peak:.0f}MB'); assert n == 21172 and peak < 100"
```

**Out of scope.** `ijson` as a backend, and streaming writers.

---

### T5 — `file_utils`

**Goal.** The readers and writers every pipeline stage needs, with atomic JSON writes.

**Context.** `$VAT/voice_agent_toolkit/file_utils.py` (10 functions) is the source. The pipeline's artifacts are all JSONL, so `read_jsonlines`/`write_jsonlines` come along; the spec did not list them. `write_json` must be atomic — the spec calls for temp-file-then-`os.replace` so a crash mid-write cannot truncate an existing artifact, which `$VAT`'s version does not do.

**Relevant files.** `$VAT/voice_agent_toolkit/file_utils.py:31,50,60,84,96,106`; spec requirement 8.

**Proposed approach.** Copy `read_txt`, `read_json`, `read_jsonlines`, `write_jsonlines`, and `create_parent_directory_from_path`. Rewrite `write_json` to write a temp file **in the destination directory** then `os.replace`. Add `read_yaml` (pyyaml is already a core dep) and `iter_json_array_file` from T4. Drop the `jsonlines` dependency in favour of stdlib line-by-line JSON so the core dependency list holds.

**Acceptance criteria.**
- `write_json` leaves the original file intact when serialization raises partway through.
- `write_json` followed by `read_json` round-trips a nested structure with non-ASCII Vietnamese text unchanged.
- `write_jsonlines` then `read_jsonlines` round-trips a list of dicts; a file with a trailing newline and a file without both read identically.
- No new core dependency beyond `json-repair` and `pyyaml`.

**Source reference.** Spec requirement 8; §Public surface.

**Verify.**
```
uv run pytest tests/test_file_utils.py -q
uv run python -c "
import json, pathlib, agent_toolkit.file_utils as fu
p = pathlib.Path('/tmp/at_atomic.json'); p.write_text('{\"keep\":1}')
class Bad:  pass
try: fu.write_json(str(p), {'x': Bad()})
except Exception: pass
assert json.loads(p.read_text()) == {'keep': 1}, 'non-atomic write truncated the file'
print('ok')"
```

**Out of scope.** S3, Postgres, Redis, Kafka, and Elasticsearch helpers — `$VAT/connections/` stays behind.

---

## Phase 3 — The LLM client works, and works reliably

**Goal.** `complete()` against any OpenAI-compatible endpoint with retry, error mapping, and concurrency control, free of host coupling. **This phase carries the plan's risk** — it is the only two-source merge.

### T6 — `llm` config, exceptions, and error mapping

**Goal.** The error taxonomy and a config layer that reads no host directory.

**Context.** Host coupling #1: `$AE/src/components/llm/config.py:22` computes `_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"` and `get_llm_config("GLM-5.1")` lowercases the model into `glm-5.1.json` under it. A library cannot reach four parents up and expect a `configs/` folder. The `LLMError` hierarchy at `$AE/exceptions.py` is 12 classes and must be preserved exactly so `agent-evaluation`'s `except` clauses keep working at v0.2.

**Relevant files.** `$AE/src/components/llm/config.py:16,22,74,114`; `$AE/src/components/llm/exceptions.py`; `$AE/src/components/llm/error_mapping.py:36,83`; spec §Breaking the host coupling #1, §Error Behavior.

**Proposed approach.** Copy `exceptions.py` and `error_mapping.py` verbatim, rewriting only the logger import. Replace the config module with `LLMConfig` plus a `ConfigResolver` Protocol and three implementations: `EnvConfigResolver` (default — `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), `DictConfigResolver` (in-memory, for tests and the pipeline), and `JsonDirConfigResolver(dir)` (preserves `agent-evaluation`'s `<dir>/<model>.json` convention). Resolution precedence: explicit argument, then injected resolver, then environment.

**Acceptance criteria.**
- All 12 exception classes exist with their current names, parents, and status codes; `LLMRateLimitError` still carries `retry_after`.
- `map_error` classifies timeouts, 429, 5xx, and connection errors as retriable, and auth, config, and other 4xx as not.
- No module references a `configs/` path or any path relative to `__file__`.
- `JsonDirConfigResolver` reproduces `get_llm_config`'s lowercase-model-to-filename behavior.
- Config resolution honours the three-level precedence, verified by a test that sets all three and asserts the argument wins.

**Source reference.** Spec requirements 5, 9; §Breaking the host coupling; §Error Behavior.

**Verify.** `uv run pytest tests/test_llm_config.py tests/test_llm_errors.py -q`

**Out of scope.** Constructing a `TrafficController` inside config resolution — that coupling is what T7 fixes.

---

### T7 — `TrafficController`, with three defects fixed

**Goal.** Concurrency gating and token-bucket throttling that survive repeated event loops and sustained throttling.

**Context.** Three defects in `$AE/src/components/llm/traffic_control.py`, all fixed here rather than carried forward. (a) `_acquire_credit` at :100 calls itself after sleeping, so sustained throttling with many waiters grows the stack one frame per retry. (b) `get_llm_config` is `@lru_cache(maxsize=None)` and builds a `TrafficController` holding an `asyncio.Semaphore` at `config.py:114`; the cache is process-global, so a second `asyncio.run()` in the same process — routine in tests and CLI stages — reuses a semaphore whose waiters belong to a closed loop. (c) `active_requests` at :110 returns `self.max_concurrency - self._semaphore._value`, reading a CPython private.

**Relevant files.** `$AE/src/components/llm/traffic_control.py:27,100,110`; `$AE/src/components/llm/config.py:114`; spec §Defects fixed during extraction.

**Proposed approach.** Rewrite `_acquire_credit` as `while True` with identical logic. Track an explicit `_in_flight` counter incremented in `__aenter__` and decremented in `__aexit__`, and have `active_requests` return it. Key the controller by `(model, id(running_loop))` and construct it lazily on first use inside a running loop, so config resolution no longer builds one.

**Acceptance criteria.**
- Concurrency never exceeds `max_concurrency` under a burst of 50 concurrent acquisitions.
- Token-bucket refill is correct under a fake clock.
- Sustained throttling adds no stack depth — a test measuring `len(inspect.stack())` inside the wait path across 200 throttled acquisitions shows no growth.
- The same cached controller is usable across two successive `asyncio.run()` calls in one process without a "future attached to a different loop" error.
- `active_requests` is correct after both normal exit and an exception inside the context manager, and reads no underscore attribute of the semaphore.

**Source reference.** Spec §Defects fixed during extraction; invariant checks under §Testing Strategy (TrafficController).

**Verify.**
```
uv run pytest tests/test_traffic_control.py -q
uv run python -c "
import asyncio
from agent_toolkit.llm import TrafficController
tc = TrafficController(max_concurrency=2, tokens_per_minute=10_000)
async def go():
    async with tc: return tc.active_requests
for _ in range(2): print(asyncio.run(go()))   # must not raise on the second run"
```

**Out of scope.** The `stream()` + `TrafficController` `AttributeError` at `$AE/factory.py:378` — `stream()` itself is v0.2, so its fix goes with it. Recorded in [v0.2](#deferred-to-v02) so it is not mistaken for an oversight.

---

### T8 — `llm.complete`

**Goal.** One async completion call with retry, error mapping, and traffic control, configured by argument, resolver, or environment.

**Context.** Host coupling #2: `$AE/src/components/llm/factory.py:35` does `from src.dependencies import settings` then `DEFAULT_MAX_RETRIES = settings.retry.max_retries`. Coupling #3: `$AE/executors.py:12` and `traffic_control.py:22` import the host's logger. `complete()` at `factory.py:161` already uses `async with _traffic_ctrl:` correctly and is the pattern to keep.

**Relevant files.** `$AE/src/components/llm/factory.py:35,63,113,161`; `$AE/src/components/llm/executors.py:19,37,49`; `$AE/src/components/llm/utils.py:29,43,57,71`; spec §Breaking the host coupling #2, #3.

**Proposed approach.** Copy `complete()` and `sdk_complete()`, replacing the settings import with an explicit `RetryPolicy` dataclass (`max_retries=8`, `base_delay=5.0`, `exponential_backoff=True` — the current `LLMRetryConfig` defaults) passed per call or set once via `set_default_retry_policy()`. Rewrite both logger imports. Bring `extract_response_content` and `_is_thinking_model` from `utils.py`/`executors.py`. Use `respx` to mock the endpoint in tests.

**Acceptance criteria.**
- `complete()` succeeds against a mocked OpenAI-compatible endpoint and returns the message content as a string.
- Retry-then-succeed, retry-exhausted, and non-retriable-fails-fast all behave per the taxonomy from T6; a 429 carrying `retry_after` raises the computed backoff floor to at least that value.
- `CancelledError`, `KeyboardInterrupt`, and `GeneratorExit` are never retried.
- The concurrency slot is acquired and released around every attempt, including failed ones.
- `extract_response_content` returns `""` rather than a repr for a complex SDK object.
- Importing `agent_toolkit.llm` without the `llm` extra installed raises a clear `ImportError` naming the extra.

**Source reference.** Spec requirements 6, 9; §Error Behavior (retry classification).

**Verify.** `uv run pytest tests/test_llm_complete.py -q`

**Out of scope.** `stream`, `complete_with_tools`, sync wrappers, and any binding other than the OpenAI-compatible path.

---

## Phase 4 — The jury's requirements are met

**Goal.** A caller can identify a model's family, count tokens before spending them, and force a schema-conforming response. These three unblock pipeline requirements 18–22 and 26.

### T9 — Model metadata: family, capabilities, token count

**Goal.** Facts about a model, harvested from `$VAT` with two staleness bugs fixed.

**Context.** The pipeline's jury requires ≥3 jurors from ≥3 distinct families and forbids any juror from the family that labelled the corpus. `$VAT/llm/llm_utils.py:68` has `check_llm_model_family`, and `$VAT/llm/constants.py` has the pattern tables. Two are stale in ways that hit us directly: `REASONING_LLMS_PATTERNS` contains `^glm-4.*`, which does **not** match `glm-5.1` — the guided-validation default generator — so reasoning-tag handling silently does not apply; and `NON_NATIVE_FC_LLMS_PATTERNS` contains `^gemma-3-.*`, which does **not** match `gemma-4-31B-it` — the model that labelled 67.3% of the corpus — so gemma-4 is treated as having native function calling. `$VAT` updated `^gpt-5.*` and missed both.

**Relevant files.** `$VAT/voice_agent_toolkit/llm/constants.py`; `$VAT/voice_agent_toolkit/llm/llm_utils.py:44,56,68,86`; `$VAT/voice_agent_toolkit/llm/openai_client.py:421` (`count_tokens`); `$AE/src/components/llm/utils.py:29`.

**Proposed approach.** New module `llm/model_meta.py` holding the pattern tables and `model_family(name)`, `supports_reasoning(name)`, `supports_native_tool_calling(name)`, `count_tokens(messages, model)`. Fix the two patterns to match current model lines and add a test asserting each specific case. Put `tiktoken` behind the `llm` extra.

**Acceptance criteria.**
- `model_family` returns distinct values for the gemma, glm, gpt, qwen, and deepseek lines, and a documented fallback for an unknown name rather than raising.
- `supports_reasoning("glm-5.1")` is `True` — the regression test for the first stale pattern.
- `supports_native_tool_calling("gemma-4-31B-it")` is `False` — the regression test for the second.
- `count_tokens` returns a positive integer for a Vietnamese message list and is within 10% of the provider's reported `prompt_tokens` on one recorded real response.
- Adding a model line means editing one table and adding one test case; nothing else changes.

**Source reference.** Consumer requirements: `../sft-dataset-pipeline/spec.md` requirements 19, 20, 26.

**Verify.**
```
uv run pytest tests/test_model_meta.py -q
uv run python -c "
from agent_toolkit.llm.model_meta import model_family, supports_reasoning, supports_native_tool_calling
assert supports_reasoning('glm-5.1'), 'stale glm pattern'
assert not supports_native_tool_calling('gemma-4-31B-it'), 'stale gemma pattern'
print(model_family('gemma-4-31B-it'), model_family('glm-5.1'))"
```

**Out of scope.** `MODEL_THOUGHT_TOKENS` per-family thought-token handling beyond what `clean_thinking_tags` already covers.

---

### T10 — Structured output

**Goal.** `complete_structured()` returning a value validated against a JSON schema, with repair on near-misses.

**Context.** Every juror must return a JSON array of tool names drawn from that record's catalog, and an invalid vote must become a clean abstention rather than a truncated partial vote. `$VAT/llm/llm_utils.py:242,319` has `validate_and_fix_structured_output`, `$VAT/openai_client.py:202` has `_sanitize_schema_for_openai_strict`, and `:121` has `_schema_to_guided_grammar` — the last constrains a locally-served model at decode time, which drives the invalid-vote rate toward zero instead of validating after the fact. `$AE` has none of this.

**Relevant files.** `$VAT/voice_agent_toolkit/llm/llm_utils.py:147,207,242,319`; `$VAT/voice_agent_toolkit/llm/openai_client.py:102,121,202,240,465,548`.

**Proposed approach.** `complete_structured(prompt, schema, *, mode="auto", …)` layering three strategies: native structured output where the provider supports it, guided grammar where a local endpoint accepts it, and prompt-plus-validate otherwise. Parse with `extract_json_from_text` from T3, then `validate_and_fix_structured_output`. Return `(value, ValidationInfo)` so the caller can tell a clean parse from a repaired one from a failure — the pipeline needs that distinction to record abstentions.

**Acceptance criteria.**
- A response that is a clean JSON array validates and reports `repaired=False`.
- A fenced, prose-wrapped, or trailing-comma response validates and reports `repaired=True`.
- A response violating the schema — wrong type, or an enum value outside the allowed set — returns a failure the caller can distinguish from success, and never a partially-parsed value.
- An enum-constrained schema built from a per-call list of allowed strings rejects any value outside it, which is what makes a juror vote structurally catalog-bounded.
- `mode="grammar"` against an endpoint that rejects grammars falls back rather than failing the call, and records which strategy was used.

**Source reference.** D3; consumer requirement `../sft-dataset-pipeline/spec.md` requirement 18.

**Verify.** `uv run pytest tests/test_structured_output.py -q` — port the relevant cases from `$VAT/tests/test_structured_output.py` (17 KB) as the starting corpus.

**Out of scope.** Streaming structured output; tool-call-shaped structured responses (that is `complete_with_tools`, v0.2).

---

## Phase 5 — v0.1 is installable by the pipeline

**Goal.** The pipeline can add one dependency and start on stage 0.

### T11 — Cut v0.1.0 and verify the consumer contract

**Goal.** A wheel installed into the pipeline's environment, with every symbol the pipeline imports confirmed present and working.

**Context.** The pipeline pins Python 3.12.14 and needs exactly: `iter_json_array_file`, `read_json`, `read_jsonlines`, `write_jsonlines`, `write_json`, `slot_filling`, `extract_json_from_text`, `normalize_text`, `compute_hash`, `complete`, `complete_structured`, `model_family`, `count_tokens`, `TrafficController`, and the `LLMError` hierarchy. Registry publishing is deferred, so the pipeline installs from a local path or a git ref — which also removes the registry-credentials assumption from the critical path.

**Relevant files.** `../sft-dataset-pipeline/spec.md` §Versions; spec §Public surface.

**Proposed approach.** Set `__version__ = "0.1.0"`, tag it, build the wheel. Write a README covering install (both extras), the three config resolvers, and a short example per module. Install into a clean 3.12 virtualenv and run a consumer smoke script importing all fifteen symbols. Record the v0.2 backlog in the README.

**Acceptance criteria.**
- The wheel installs into a clean 3.12 venv; core-only install succeeds and `import agent_toolkit.llm` then raises an `ImportError` naming the `llm` extra.
- All fifteen consumer symbols import and each is exercised once by the smoke script.
- `agent_toolkit.__version__ == "0.1.0"` and the git tag matches.
- README documents the deferred surface so a reader does not expect `stream` or `complete_with_tools`.

**Source reference.** Spec requirements 1, 3; §Versions.

**Verify.**
```
uv build
python3.12 -m venv /tmp/at-core && /tmp/at-core/bin/pip install -q dist/*.whl
/tmp/at-core/bin/python -c "import agent_toolkit.string_utils, agent_toolkit.json_utils; print('core ok')"
/tmp/at-core/bin/python -c "
try: import agent_toolkit.llm; raise SystemExit('should have failed')
except ImportError as e: assert 'llm' in str(e).lower(); print('extra gate ok')"
python3.12 -m venv /tmp/at-full && /tmp/at-full/bin/pip install -q "dist/*.whl[llm]"
/tmp/at-full/bin/python tests/consumer_smoke.py
```

**Out of scope.** Publishing to the internal registry, and the `agent-evaluation` migration.

---

## Deferred to v0.2

Each item names why it is not in v0.1.

| Item | Why deferred |
|---|---|
| `stream()` and its `TrafficController` fix (`$AE/factory.py:378` calls `_wait_for_token()`, which does not exist, so **every** streaming call raises `AttributeError`) | No v0.1 consumer streams. The jury, the scrubber, and the question generator are all non-streaming. The fix ships with the feature. |
| `complete_with_tools()` | Jurors return JSON text, not tool calls. |
| Full behavioural parity with `voice-agent-toolkit` 0.2.24 | Parity exists to make `agent-evaluation`'s swap provably transparent. DataForce is greenfield. Gates the migration, not v0.1. |
| The `agent-evaluation` migration (16 import lines, `JsonDirConfigResolver` install, deleting its local copies) | Blocked on the parity corpus above, and on the other team's schedule. |
| Registry publishing, 3.11 and 3.14 wheels | The pipeline installs from a path or git ref on 3.12. Removes the credentials assumption from the critical path. |
| Embeddings, Responses API conversion, `BaseLLMClient` provider abstraction | Present in `$VAT` and worth harvesting, but the pipeline uses local static embeddings and one OpenAI-compatible path. |
| `extract_tool_calls_from_text`, the tool protocol | Spec §Out of Scope. `$VAT/agent/tool_utils.py` is 37 KB and unread; specifying a port of unread code would be guessing. |
| `$VAT/connections/*`, `knowledge_base/`, `clustering/`, `conversation/`, `prompt_evolution/`, `state_machine_extraction/` | Out of the library's purpose, and the source of `$VAT`'s 17 hard dependencies. |
| `text_normalization/digit_processing/` | Goes to the pipeline, not here — D4. |

## Sequencing

```
T1 ─ T2 ─┬─ T3 ─┐
         ├─ T4 ─┼─ T6 ─ T7 ─ T8 ─┬─ T9 ──┐
         └─ T5 ─┘                └─ T10 ─┴─ T11
```

T3, T4, and T5 are independent of each other and can run in parallel after T2. T6→T7→T8 is a strict chain — the risk concentration, and the one place to expect surprises. T9 and T10 are independent once T8 lands; T10 depends on T3's `extract_json_from_text`.

## Open questions

None blocking. Two worth resolving before v0.2 rather than v0.1:

1. Whether `agent-evaluation` will accept `>=3.11,<4` in place of its `==3.12.3` pin — inherited from `$VAT/pyproject.toml`, not authored there. If it will not, the migration needs its own compatibility story.
2. Which internal registry group publishes this, and whether its CI credentials exist. Deferred, not resolved, because v0.1 installs from a path.
