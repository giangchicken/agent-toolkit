"""String helpers: slot filling, JSON extraction, normalization, hashing.

``slot_filling`` and ``extract_json_from_text`` are harvested from
``agent-evaluation``'s ``src/utils/string_utils.py``, which was itself a
toolkit-free copy of the originals. Behavior is preserved: ``json_repair``-based
parsing, a code-fence scan, a first/last brace then bracket scan, then a
nested-candidate sweep. ``normalize_text`` and ``compute_hash`` come from
``voice-agent-toolkit``.

Both extraction functions log at debug and return a fallback rather than raising,
on any input. That contract is deliberate: they parse model output, which is
adversarial by nature, and a caller mid-pipeline should get an empty result it
can record rather than an exception it has to catch at every call site.
"""

import hashlib
import json
import re
import unicodedata
from typing import Any

import json_repair

from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "clean_thinking_tags",
    "compute_hash",
    "extract_json_from_text",
    "normalize_text",
    "slot_filling",
]

# Narrow no-break space, no-break space, thin space. Written as escapes
# because all three are visually indistinguishable from U+0020 in source.
SPACE_UNICODES = ["\u202f", "\u00a0", "\u2009"]

# Each pass of slot_filling resolves at least one level of nesting, and the
# design it serves is two-level, so this is roughly fifty times the depth any
# real template needs. It exists to bound one case the upstream fixpoint loop
# does not terminate on: mutually referential placeholders. {"a": "{{b}}",
# "b": "{{a}}"} changes the text on every pass, so the `text == old_text` check
# never fires and the loop spins forever. Direct self-reference is fine -- the
# replacement is a no-op, so the fixpoint check catches it on the first pass.
MAX_SLOT_FILLING_PASSES = 100


def compute_hash(content: str, hash_type: str = "sha256") -> str:
    """Return a hex digest of ``content``. Unknown ``hash_type`` means sha256."""
    if hash_type == "md5":
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    elif hash_type == "sha1":
        return hashlib.sha1(content.encode("utf-8")).hexdigest()
    elif hash_type == "sha512":
        return hashlib.sha512(content.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_text(text: str, remove_tone_marks: bool = False) -> str:
    """NFKC-normalize ``text``, fold unusual spaces, and collapse whitespace.

    Digits separated by a narrow or non-breaking space are joined rather than
    space-separated, so "1 234" written with U+202F becomes "1234".
    ``remove_tone_marks`` strips Unicode combining marks, which for Vietnamese
    means dropping diacritics -- useful as a dedup key, lossy as a display form.
    """
    try:
        for unicode_space in SPACE_UNICODES:
            if unicode_space in text:
                text = re.sub(rf"(\d){unicode_space}(\d)", r"\1\2", text)
                text = text.replace(unicode_space, " ")
        text = unicodedata.normalize("NFKC", text)
        if remove_tone_marks:
            # Decompose first. The harvested version filtered combining marks
            # straight after NFKC, but NFKC *recomposes*, so for Vietnamese there
            # were no Mn characters left to find and the flag did nothing at all.
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
            text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.debug(f"normalize_text failed: {e}")
        return text


def slot_filling(
    text: str,
    key_value_mapping: dict[str, Any] | None = None,
    object_dict: dict[str, Any] | None = None,
) -> str:
    """Substitute ``{{placeholder}}`` tokens until the text stops changing.

    A value that itself contains ``{{other}}`` is resolved on a later pass, so
    nested placeholders fill correctly. ``object_dict`` values are read from a
    ``{"value": ...}`` wrapper; ``key_value_mapping`` values are used directly.
    Unknown placeholders are left untouched.

    Gives up after ``MAX_SLOT_FILLING_PASSES`` passes and returns the text as it
    stands, which bounds mutually referential placeholders.
    """
    try:
        no_change = False
        passes = 0
        while not no_change:
            if passes >= MAX_SLOT_FILLING_PASSES:
                logger.debug(
                    "slot_filling did not converge in %d passes; "
                    "placeholders may be mutually referential",
                    MAX_SLOT_FILLING_PASSES,
                )
                return text
            passes += 1
            placeholders = re.findall(r"{{(.*?)}}", text)
            old_text = text
            mapping_dict = {}
            for placeholder in placeholders:
                if placeholder in mapping_dict:
                    continue
                if isinstance(object_dict, dict) and placeholder in object_dict:
                    mapping_dict[placeholder] = object_dict[placeholder]["value"]
                elif (
                    isinstance(key_value_mapping, dict)
                    and placeholder in key_value_mapping
                ):
                    mapping_dict[placeholder] = key_value_mapping[placeholder]
            for mapping_key in mapping_dict:
                text = text.replace(
                    "{{" + mapping_key + "}}",
                    str(mapping_dict.get(mapping_key, "{{" + mapping_key + "}}")),
                )
            no_change = text == old_text
        return text
    except Exception as e:
        logger.debug(f"slot_filling failed: {e}")
        return text


def _spans_one_structure(span: str) -> bool:
    """True if ``span`` is a single brace/bracket structure rather than two.

    The first-brace-to-last-brace scan below can select a span covering two
    disjoint objects -- ``{"a": 1} and then {"b": 2}`` -- which a repair parser
    will happily turn into a single value, swallowing both and reporting one.
    This guard rejects that span so it falls through to the candidate sweep,
    which finds the two objects separately.

    Depth is counted with string contents skipped, so a brace inside a JSON
    string value cannot throw off the count.
    """
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(span):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0 and span[index + 1 :].strip():
                return False
    return True


def _parse_json_candidate(json_text: str) -> dict[str, Any] | list[Any] | None:
    """Parse one candidate, discarding a repair that changed the structure's type.

    ``json_repair`` turns ``{invalid json here}`` into ``["invalid json here}"]``
    -- a list invented from something that opened as an object. For a caller
    recording model votes that is worse than no result at all, because the
    garbage arrives looking like a real answer. So a candidate opening with
    ``{`` must parse to a dict and one opening with ``[`` must parse to a list,
    or it is discarded.
    """
    stripped = json_text.lstrip()
    try:
        parsed = json_repair.loads(json_text)
    except json.JSONDecodeError:
        return None
    if stripped.startswith("{") and isinstance(parsed, dict):
        return parsed
    if stripped.startswith("[") and isinstance(parsed, list):
        return parsed
    return None


def extract_json_from_text(
    text: str, extract_all: bool = False
) -> dict[str, Any] | list[Any] | None:
    """Extract the first JSON value from ``text`` (or all, if ``extract_all``).

    Scan order: ```json fenced blocks, then the outer first/last brace object,
    then the outer first/last bracket array, then a sweep of nested
    brace/bracket candidates. Parsing uses ``json_repair`` to tolerate minor
    malformation. Returns the parsed ``dict``/``list``, a list of them when
    ``extract_all`` is set, or ``None`` when nothing parses.
    """
    try:
        json_objects: list[Any] = []
        used_positions: set[int] = set()

        text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)

        json_code_blocks = re.finditer(r"```json(.*?)```", text, re.DOTALL)
        for match in json_code_blocks:
            json_text = match.group(1).strip()
            parsed = _parse_json_candidate(json_text)
            if parsed is not None:
                json_objects.append(parsed)
                used_positions.update(range(match.start(), match.end()))

        first_brace_pos = text.find("{")
        last_brace_pos = text.rfind("}")
        if (
            first_brace_pos != -1
            and last_brace_pos != -1
            and first_brace_pos < last_brace_pos
            and not any(
                pos in used_positions
                for pos in range(first_brace_pos, last_brace_pos + 1)
            )
            and _spans_one_structure(text[first_brace_pos : last_brace_pos + 1])
        ):
            json_text = text[first_brace_pos : last_brace_pos + 1]
            parsed = _parse_json_candidate(json_text)
            if parsed is not None:
                json_objects.append(parsed)
                used_positions.update(range(first_brace_pos, last_brace_pos + 1))

        first_bracket_pos = text.find("[")
        last_bracket_pos = text.rfind("]")
        if (
            first_bracket_pos != -1
            and last_bracket_pos != -1
            and first_bracket_pos < last_bracket_pos
            and not any(
                pos in used_positions
                for pos in range(first_bracket_pos, last_bracket_pos + 1)
            )
            and _spans_one_structure(text[first_bracket_pos : last_bracket_pos + 1])
        ):
            json_text = text[first_bracket_pos : last_bracket_pos + 1]
            parsed = _parse_json_candidate(json_text)
            if parsed is not None:
                json_objects.append(parsed)
                used_positions.update(range(first_bracket_pos, last_bracket_pos + 1))

        all_candidates = []

        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL):
            if not any(
                pos in used_positions for pos in range(match.start(), match.end())
            ):
                all_candidates.append((match.start(), match.end(), match.group()))

        for match in re.finditer(
            r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text, re.DOTALL
        ):
            if not any(
                pos in used_positions for pos in range(match.start(), match.end())
            ):
                all_candidates.append((match.start(), match.end(), match.group()))

        all_candidates.sort(key=lambda x: (x[0], -x[1]))

        for start, end, json_text in all_candidates:
            if any(pos in used_positions for pos in range(start, end)):
                continue
            parsed = _parse_json_candidate(json_text)
            if parsed is not None:
                json_objects.append(parsed)
                used_positions.update(range(start, end))

        if extract_all:
            return json_objects
        return json_objects[0] if json_objects else None
    except Exception as e:
        logger.debug(f"extract_json_from_text failed: {e}")
        return None


def clean_thinking_tags(
    content: str,
    binding: str | None = None,
    model: str | None = None,
) -> str:
    """Remove ``<think>`` blocks from model output.

    ``binding`` and ``model`` are accepted and ignored. They are part of the
    signature at the harvested call sites, and keeping them is what makes the
    ``agent-evaluation`` migration an import-line change.
    """
    if not content:
        return ""

    pattern = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(pattern, "", content)
    return cleaned.strip()
