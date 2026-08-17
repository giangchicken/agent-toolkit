"""A schema-conforming response, plus how much work it took to get one.

The SFT pipeline's jury needs a juror to return a JSON array of tool names drawn
from that record's catalog, and an invalid vote must become a clean *abstention*
-- not a partial vote (requirement 18 of ``docs/sft-dataset-pipeline/spec.md``).
That last clause is the whole design: a value either conforms to the schema and
comes back, or it does not and comes back as ``None`` with the reason. Nothing in
between is returned, because a half-parsed vote is indistinguishable from a real
one once it is written to disk.

Three strategies, tried in an order set by ``mode``:

``native``
    ``response_format={"type": "json_schema", ...}``, the OpenAI-compatible
    parameter. Constrains decoding where the provider implements it.
``grammar``
    ``extra_body={"guided_json": schema}``, vLLM's decode-time constraint. This
    is what drives the invalid-vote rate toward zero on a self-hosted endpoint,
    because the tokens that would break the schema are never sampled.
``prompt``
    The schema appended to the prompt as an instruction. Works everywhere,
    constrains nothing, and is the fallback for the two above.

**Validation runs on the response text whichever strategy produced it.** The
constraint parameters are an optimization; this module's guarantee comes from
validating afterwards. That is what makes falling back safe.

**Measured warning: for the jury's own schema, constrained decoding is worse
than prompting.** The plan expects guided decoding to drive the invalid-vote rate
toward zero. Against a self-hosted vLLM endpoint serving ``gemma-4-31B-it``, with
the schema ``{"type": "array", "items": {"type": "string", "enum": [...]}}``, one
prompt asking for the tools needed to send a confirmation SMS:

=========================  ================================================
strategy                   result
=========================  ================================================
``prompt``                 ``["send_sms"]`` -- correct
``grammar`` / ``native``   ``"send_sms"`` repeated 20+ times until
                           ``max_tokens``, truncated, unparseable
``grammar`` + ``maxItems`` ``["send_sms", "send_sms"]`` -- schema-valid
=========================  ================================================

Nothing in the grammar forbids taking the same ``enum`` branch again, so an
unbounded array degenerates into repetition. ``maxItems`` compiles and merely
truncates the repetition, which is the worse outcome of the two: a duplicate vote
passes validation and is indistinguishable downstream from one the model meant.
``uniqueItems`` would fix it, and that endpoint answers **HTTP 500** to it -- a
retriable status, so the full retry policy is spent before the call fails, and no
fallback happens because a 500 is not a rejected parameter.

The second measurement points the same way. Given a catalog that excludes the
answer the model wants, ``prompt`` returned ``[]`` -- the abstention requirement
18 asks for -- while ``grammar`` and ``native`` returned
``["check_weather", "check_weather"]``: forced into the enum, semantically empty,
and valid. **Constrained decoding does not remove invalid votes; it removes
*detectable* ones.** Prefer ``mode="prompt"`` for jury work and keep the
validation result, which is the part that carries information.

Harvested from ``voice-agent-toolkit``, where four pieces existed and none is
carried over unchanged.

**``validate_and_fix_structured_output`` is not ported.** It walks a schema's
``required`` list and builds a dict, so it cannot validate a top-level array --
the one shape requirement 18 asks for. It ignores ``enum`` entirely, so the
catalog bound that makes a vote meaningful was never checked. It is annotated
``-> Dict`` and returns ``json.dumps(...)``, a ``str``. And its repair sets a bad
field to ``None`` and reports success, which is exactly the fabricated vote the
requirement forbids. ``jsonschema`` replaces it.

**``_schema_to_guided_grammar`` is not ported.** Its 55 lines emit
``root ::= "{" ... "}"``: the root is always an object, so it cannot express the
pipeline's array either. vLLM's ``guided_json`` takes the schema directly. The
harvested reason for hand-building EBNF -- clamping whitespace, because xgrammar
would emit long runs of it -- is a token-cost problem with a server-side knob
(``guided_whitespace_pattern``), not a correctness one.

**``_sanitize_schema_for_openai_strict`` is not ported, because ``strict`` is not
set.** OpenAI's strict Structured Outputs requires an object at the root, so the
pipeline's array-of-tool-names cannot use it at all; the harvested chat path did
not set ``strict`` either. Note that ``nullable: true``, which the harvested
schema builder emits, is OpenAPI and not JSON Schema -- ``jsonschema`` ignores it
and will reject ``None``. Write ``{"type": ["string", "null"]}``.

**The fallback condition is narrowed.** The harvest wrapped the call in a bare
``except Exception`` and retried without ``response_format``, which also
swallowed auth failures and timeouts and then re-ran them identically. Here only
a client error that means "this provider does not accept that parameter" falls
through; everything else propagates as itself.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

import jsonschema
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

from agent_toolkit.llm.exceptions import LLMAPIError, LLMConfigError
from agent_toolkit.llm.factory import complete
from agent_toolkit.logging import get_logger
from agent_toolkit.string_utils import extract_json_from_text

logger = get_logger(__name__)

__all__ = ["ValidationInfo", "complete_structured"]

Mode = Literal["auto", "grammar", "prompt"]
Strategy = Literal["native", "grammar", "prompt"]

# ``auto`` is native-then-prompt rather than something cleverer because there is
# no capability signal for structured output the way there is for tool calling:
# a model name does not tell you whether the endpoint serving it implements
# ``response_format``. Trying it and reading the rejection does.
_CHAINS: dict[str, tuple[Strategy, ...]] = {
    "auto": ("native", "prompt"),
    "grammar": ("grammar", "prompt"),
    "prompt": ("prompt",),
}

# "You sent a parameter I do not implement" -- vLLM and OpenAI both answer 400,
# some gateways answer 422. Any other status is a real failure: a 401 would fail
# the fallback identically, and a 500 is the retry policy's business.
_UNSUPPORTED_STATUSES = (400, 422)

_SCHEMA_NAME = "structured_output"

_INSTRUCTION = (
    "Respond with JSON only, conforming to this JSON Schema:\n"
    "{schema}\n"
    "Output the JSON value alone -- no explanation, no code fence."
)

# Distinguishes "nothing parsed" from a response that is the JSON value ``null``.
_MISSING = object()


@dataclass(frozen=True)
class ValidationInfo:
    """Why a :func:`complete_structured` result is what it is.

    Attributes:
        ok: The value conforms to the schema. ``False`` means the returned value
            is ``None`` and this record is an abstention.
        strategy: Which of the three actually produced ``raw`` -- the requested
            mode's first choice, or whatever it fell back to.
        repaired: The response was not valid JSON on its own and had to go
            through ``extract_json_from_text``: fenced, prose-wrapped, trailing
            comma, or unparseable. A caller tracking model quality wants this
            separately from ``ok``.
        raw: The response text. The only evidence of *what* failed, and
            unrecoverable once discarded, so it is kept even on success.
        error: One-line reason when ``ok`` is False, else ``None``.
    """

    ok: bool
    strategy: Strategy
    repaired: bool
    raw: str
    error: str | None = None


def _apply(
    strategy: Strategy, prompt: str, schema: dict[str, Any], kwargs: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """The prompt and call kwargs ``strategy`` needs, leaving ``kwargs`` alone."""
    if strategy == "native":
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": _SCHEMA_NAME, "schema": schema},
        }
        return prompt, {**kwargs, "response_format": response_format}

    if strategy == "grammar":
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["guided_json"] = schema
        return prompt, {**kwargs, "extra_body": extra_body}

    instruction = _INSTRUCTION.format(schema=json.dumps(schema, ensure_ascii=False))
    messages = kwargs.get("messages")
    if messages:
        # ``complete`` ignores ``prompt`` when ``messages`` is set, so appending
        # the instruction to the prompt would send the schema nowhere.
        appended = [*messages, {"role": "user", "content": instruction}]
        return prompt, {**kwargs, "messages": appended}
    return f"{prompt}\n\n{instruction}", kwargs


def _parse(text: str) -> tuple[Any, bool]:
    """The JSON value in ``text``, and whether getting it needed repair."""
    try:
        return json.loads(text), False
    except ValueError:
        pass
    extracted = extract_json_from_text(text)
    return (_MISSING if extracted is None else extracted), True


async def _request(
    prompt: str,
    schema: dict[str, Any],
    chain: tuple[Strategy, ...],
    kwargs: dict[str, Any],
) -> tuple[str, Strategy]:
    *fallbacks, final = chain
    for index, strategy in enumerate(fallbacks):
        call_prompt, call_kwargs = _apply(strategy, prompt, schema, kwargs)
        try:
            return await complete(prompt=call_prompt, **call_kwargs), strategy
        except LLMAPIError as exc:
            if exc.status_code not in _UNSUPPORTED_STATUSES:
                raise
            logger.warning(
                "%s structured output rejected (HTTP %s), falling back to %s: %s",
                strategy,
                exc.status_code,
                chain[index + 1],
                exc.message,
            )
    call_prompt, call_kwargs = _apply(final, prompt, schema, kwargs)
    return await complete(prompt=call_prompt, **call_kwargs), final


async def complete_structured(
    prompt: str,
    schema: dict[str, Any],
    *,
    mode: Mode = "auto",
    **kwargs: Any,
) -> tuple[Any, ValidationInfo]:
    """Complete ``prompt`` and return a value validated against ``schema``.

    Args:
        prompt: The user prompt. The ``prompt`` strategy appends the schema to it.
        schema: A JSON Schema. Validated as a schema before any request is sent,
            so a malformed one costs no tokens.
        mode: ``"auto"`` tries ``response_format`` then falls back to prompting;
            ``"grammar"`` tries vLLM's ``guided_json`` then falls back to
            prompting; ``"prompt"`` only prompts. For an unbounded array of enum
            values, ``"prompt"`` is the one that measured correct -- see this
            module's docstring before choosing either of the others.
        **kwargs: Forwarded to :func:`agent_toolkit.llm.complete` -- ``model``,
            ``system_prompt``, ``retry``, ``temperature``, and the rest.

    Returns:
        ``(value, info)``. ``value`` is the validated JSON value, or ``None``
        when ``info.ok`` is False. Never a partially valid value.

    Raises:
        LLMConfigError: ``mode`` is not one of the three, or ``schema`` is not a
            valid JSON Schema.
        LLMError: Any provider failure, after the retry policy is exhausted. A
            provider *rejecting* the constraint parameter is not a failure; it
            falls back. A response that violates the schema is not a failure
            either -- it is a result with ``ok=False``.
    """
    chain = _CHAINS.get(mode)
    if chain is None:
        raise LLMConfigError(
            f"unknown mode {mode!r}; expected one of {sorted(_CHAINS)}"
        )

    validator_cls = validator_for(schema)
    try:
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise LLMConfigError(f"invalid JSON schema: {exc.message}") from exc

    text, strategy = await _request(prompt, schema, chain, kwargs)
    value, repaired = _parse(text)

    if value is _MISSING:
        return None, ValidationInfo(
            ok=False,
            strategy=strategy,
            repaired=repaired,
            raw=text,
            error="no JSON object or array in the response",
        )

    failure = best_match(validator_cls(schema).iter_errors(value))
    if failure is not None:
        return None, ValidationInfo(
            ok=False,
            strategy=strategy,
            repaired=repaired,
            raw=text,
            error=f"{failure.json_path}: {failure.message}",
        )

    return value, ValidationInfo(
        ok=True, strategy=strategy, repaired=repaired, raw=text
    )
