"""logic. The decisions taken over the tables in :mod:`agent_toolkit.lexicon`."""

import hashlib
import json
import re
import unicodedata
from itertools import takewhile
from typing import Any

import json_repair

from agent_toolkit.lexicon import (
    NAME_TITLES,
    OTP_CUES,
    SPACE_UNICODES,
    SPOKEN_AT,
    SPOKEN_DIGITS,
    SPOKEN_DOT,
    THINKING_MARKERS,
)
from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "compute_hash",
    "email_detection_by_rules",
    "extract_json_from_text",
    "name_detection_by_rules",
    "normalize_text",
    "otp_detection_by_rules",
    "phone_number_detection_by_rules",
    "slot_filling",
    "split_thinking",
]


MAX_SLOT_FILLING_PASSES = 100


def compute_hash(content: str, hash_type: str = "sha256") -> str:
    if hash_type == "md5":
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    elif hash_type == "sha1":
        return hashlib.sha1(content.encode("utf-8")).hexdigest()
    elif hash_type == "sha512":
        return hashlib.sha512(content.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_text(text: str, remove_tone_marks: bool = False) -> str:
    try:
        for unicode_space in SPACE_UNICODES:
            if unicode_space in text:
                text = re.sub(rf"(\d){unicode_space}(\d)", r"\1\2", text)
                text = text.replace(unicode_space, " ")
        text = unicodedata.normalize("NFKC", text)
        if remove_tone_marks:
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


def split_thinking(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    lowered = text.lower()
    for start, end in THINKING_MARKERS:
        end_at = lowered.find(end)
        if end_at != -1:
            cut = end_at + len(end)
            return text[:cut].strip(), text[cut:].strip()
        start_at = lowered.find(start)
        if start_at != -1:
            return text[start_at:].strip(), ""
    return "", text


def phone_number_detection_by_rules(text: str, language: str = "vi") -> list[str]:
    said = SPOKEN_DIGITS[language]
    return re.findall(
        r"(?<!\w)\+?\d(?:[\s.-]?\d){8,10}(?!\w)"
        rf"|(?:{said})(?:[\s.,]+(?:{said})){{8,10}}",
        text,
        re.IGNORECASE,
    )


def email_detection_by_rules(text: str, language: str = "vi") -> list[str]:
    at, dot = SPOKEN_AT[language], SPOKEN_DOT[language]
    return re.findall(
        r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
        rf"|(?:[\w.+-]+\s+){{1,4}}(?:{at})(?:\s+[\w-]+)+?"
        rf"(?:\s+(?:{dot})\s+[\w-]+)+",
        text,
        re.IGNORECASE,
    )


def otp_detection_by_rules(text: str, language: str = "vi") -> list[str]:
    cue, said = OTP_CUES[language], SPOKEN_DIGITS[language]
    return re.findall(
        rf"(?:{cue}).{{0,20}}?"
        rf"(\d(?:[\s.-]?\d){{3,7}}|(?:{said})(?:[\s.,]+(?:{said})){{3,7}})",
        text,
        re.IGNORECASE,
    )


def name_detection_by_rules(text: str, language: str = "vi") -> list[str]:
    title = NAME_TITLES[language]
    names = []
    for cue in re.finditer(rf"(?<!\w)(?:{title})[\s,.:]+", text, re.IGNORECASE):
        after = re.findall(r"[^\W\d_]+", text[cue.end() : cue.end() + 60])[:4]
        found = " ".join(takewhile(lambda word: word[:1].isupper(), after))
        if found:
            names.append(found)
    return names
