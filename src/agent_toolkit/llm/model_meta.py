"""Facts about a model name: which family, which capabilities, how many tokens.

The SFT pipeline's jury needs all three (requirements 19, 20 and 26 of
``docs/sft-dataset-pipeline/spec.md``): a panel must draw at least three jurors
from at least three *distinct* families, no juror may come from the family that
labelled the corpus, and a stage must estimate its token cost before spending it.

Harvested from ``voice-agent-toolkit``'s ``llm/constants.py`` and
``llm/llm_utils.py``, with three differences.

**The family function knew five families and needed eight.**
``check_llm_model_family`` recognised gpt, qwen, llama/vicuna and gemini and
returned ``"unknown"`` for everything else -- which put gemma, glm and deepseek
in one bucket. Since the jury requirement counts *distinct* families, a panel of
gemma + glm + deepseek read as one family rather than three, and the
corpus-labelling family requirement 20 exists to exclude -- ``gemma-4-31B-it``,
67.3% of the corpus -- had no name at all. Those three lines are in the table now.

**``^glm-5.*`` was missing from both capability tables.** The reasoning table's
copy in ``agent-evaluation`` had already been patched and
``voice-agent-toolkit``'s had not, which is what keeping two copies of one table
costs; the tool-calling table was unpatched in both, so ``glm-5.1`` -- the
pipeline's default generator -- was reported as having no native tool calling.
Both are fixed here, and there is one copy.

**The unread denylist is not carried over.** ``NON_NATIVE_FC_LLMS_PATTERNS``
existed in ``constants.py`` with no reader anywhere in the package:
``check_native_tool_calling_capability`` consults the allowlist only, and an
unlisted model correctly falls back to prompt-based tool calling. A second table
that is allowed to disagree with the first is how the first one drifted.

``count_tokens`` is an estimate, and for Vietnamese a poor one -- see its
docstring for drift measured against a real provider.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

import tiktoken

__all__ = [
    "FAMILY_MARKERS",
    "NATIVE_TOOL_CALLING_PATTERNS",
    "REASONING_PATTERNS",
    "UNKNOWN_FAMILY",
    "count_tokens",
    "model_family",
    "supports_native_tool_calling",
    "supports_reasoning",
]

UNKNOWN_FAMILY = "unknown"

# (family, marker) in priority order, matched as substrings of the lowercased
# name. Substrings rather than anchored patterns because a family is a property
# of the whole name, not of its version prefix -- that is the harvested
# function's ``"gpt" in llm_name`` shape.
#
# First match wins, which decides the one genuinely ambiguous case:
# ``deepseek-r1-distill-qwen-32b`` is deepseek here, following the convention of
# naming the lineage first. It is a Qwen base distilled from R1, so its errors
# correlate with both lines; a caller who cares about shared architecture rather
# than shared training lineage should swap these two rows.
FAMILY_MARKERS: list[tuple[str, str]] = [
    ("gemma", "gemma"),
    ("glm", "glm"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("gpt", "gpt"),
    ("gemini", "gemini"),
    ("llama", "llama"),
    ("llama", "vicuna"),
]

# Anchored, unlike the family markers, because these separate *versions* of one
# line: ``^glm-4.*`` must not answer for ``glm-5.1``, and the two years it did
# are why both tables below gained a ``glm-5`` row.
REASONING_PATTERNS = [
    r"^qwen3-.*",
    r"^qwen3\..*",
    r"^glm-4.*",
    r"^glm-5.*",
    r"^gpt-oss-.*",
    r"^gpt-5.*",
    r"^deepseek-.*",
]

NATIVE_TOOL_CALLING_PATTERNS = [
    r"^glm-4.*",
    r"^glm-5.*",
    r"^gpt-4.*",
    r"^gpt-5.*",
]

# One message costs this much before its content: the chat template's role
# header and turn delimiters. The OpenAI cookbook's approximation, kept.
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
_CHAT_FORMAT_OVERHEAD = 4


def model_family(name: str) -> str:
    """The model line ``name`` belongs to, or ``"unknown"``.

    Never raises. Note that ``"unknown"`` is a family like any other to a caller
    counting distinct families, so two names this table does not recognise will
    read as one family -- a jury assembled from unrecognised names is not proved
    diverse, it is unmeasured.
    """
    lowered = name.lower()
    for family, marker in FAMILY_MARKERS:
        if marker in lowered:
            return family
    return UNKNOWN_FAMILY


def supports_reasoning(name: str) -> bool:
    """Whether ``name`` emits reasoning, either as tags or a ``reasoning`` field.

    An unlisted model is False, so a caller strips no tags and reads no
    ``reasoning`` field for it. Being wrong that way leaves reasoning text in the
    response; being wrong the other way would discard part of a real answer.
    """
    lowered = name.lower()
    return any(re.match(pattern, lowered) for pattern in REASONING_PATTERNS)


def supports_native_tool_calling(name: str) -> bool:
    """Whether ``name`` accepts an OpenAI-style ``tools`` parameter.

    An allowlist: an unlisted model is False and gets prompt-based tool calling,
    which works everywhere. Defaulting the other way would send ``tools`` to a
    model that silently ignores it and then wait for a tool call that never comes.
    """
    lowered = name.lower()
    return any(re.match(pattern, lowered) for pattern in NATIVE_TOOL_CALLING_PATTERNS)


def count_tokens(
    messages: Sequence[Mapping[str, Any]], model: str | None = None
) -> int:
    """Estimate the prompt tokens ``messages`` will cost on ``model``.

    Four tokens per message plus the encoded length of every value in it, using
    tiktoken's encoding for ``model`` or ``cl100k_base`` when it has none.

    **For a model tiktoken does not know this is rough, and on Vietnamese it is
    very rough.** Measured against the ``prompt_tokens`` that ``gemma-4-31B-it``
    reported for four real requests:

    ==============  ========  =========  =====
    prompt          reported  estimated  drift
    ==============  ========  =========  =====
    8 chars, VI          15         10    -33%
    2 messages, VI       43         64    +49%
    4 messages, VI      153        251    +64%
    2 messages, EN       37         29    -22%
    ==============  ========  =========  =====

    Two errors of opposite sign, so no single correction factor removes them:
    ``cl100k_base`` splits Vietnamese diacritics into far more tokens than
    Gemma's vocabulary does, while four tokens per message understates Gemma's
    chat template. Size a request with this; account for what it actually cost
    with the ``usage`` its response reports.
    """
    try:
        encoding = tiktoken.encoding_for_model(model) if model else None
    except KeyError:
        encoding = None
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for message in messages:
        total += _CHAT_FORMAT_OVERHEAD
        for value in message.values():
            if value:
                total += len(encoding.encode(str(value)))
    return total
